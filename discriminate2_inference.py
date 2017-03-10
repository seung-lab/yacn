from __future__ import print_function
import numpy as np
import os
from datetime import datetime
import math
import itertools
import pprint
from convkernels3d import *
from activations import *
from loss_functions import *
import discrim_net
import os
from datetime import datetime
from experiments import save_experiment, repo_root
import random

import tensorflow as tf
from tensorflow.python.client import timeline

from utils import *
from dataset import MultiDataset
import dataset
#import pythonzenity

class DiscrimModel(Model):
	def __init__(self, patch_size, 
				 devices, name=None):

		self.name=name
		self.summaries = []
		self.devices = devices
		self.patch_size = patch_size
		self.padded_patch_size = (1,) + patch_size + (1,)

		patchx,patchy,patchz = patch_size

		config = tf.ConfigProto(
			allow_soft_placement=True,
			gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True),
			#log_device_placement=True,
		)
		self.sess = tf.Session(config=config)
		self.run_metadata = tf.RunMetadata()

		with tf.name_scope('params'):
			self.step=tf.Variable(0)
			discrim, reconstruct = discrim_net.make_forward_net(patch_size,1,1)
			self.discrim = discrim

		#for some reason we need to initialize here first... Figure this out!
		init = tf.global_variables_initializer()
		self.sess.run(init)

		var_list = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope='params')

		init = tf.global_variables_initializer()
		self.sess.run(init)

		self.saver = tf.train.Saver(var_list=var_list)
	
	#plan: assign to each object the magnitude of the max value of the error detector in the window.
	#vol should be machine_labels of size [1,X,Y,Z,1]
	def inference(self, machine_labels, human_labels, samples):
		N=samples.shape[0]
		initializer_feed_dict={}
		ret = Volume(static_constant_variable(np.zeros_like(machine_labels,dtype=np.float32),initializer_feed_dict), self.padded_patch_size)
		machine_labels = Volume(static_constant_variable(machine_labels,initializer_feed_dict), self.padded_patch_size)
		human_labels = Volume(static_constant_variable(human_labels,initializer_feed_dict), self.padded_patch_size)
		focus_inpt = tf.placeholder(shape=(3,),dtype=tf.int32)
		focus=tf.concat([[0],focus_inpt,[0]],0)

		machine_labels_glimpse = equal_to_centre(machine_labels[focus])
		human_labels_glimpse = human_labels[focus]

		discrim_tower = self.discrim(machine_labels_glimpse)
		i=3
		ds_shape = static_shape(discrim_tower[i])
		print(ds_shape)
		expander = compose(*reversed(discrim_net.range_expanders[0:i]))
		otpt = upsample_mean(tf.nn.sigmoid(discrim_tower[i]), self.padded_patch_size, expander)
		
		errors = localized_errors(machine_labels_glimpse, human_labels_glimpse, ds_shape = ds_shape, expander=expander)
		test_err = upsample_mean(errors, self.padded_patch_size, expander)
		test_err_mag = tf.reduce_sum(test_err)
		"""
		test_err = tf.ones_like(machine_labels_glimpse)
		test_err_mag=tf.constant(1)
		"""

		it = ret.__setitem__(focus,tf.maximum(test_err * machine_labels_glimpse,ret[focus]))

		self.sess.run(tf.global_variables_initializer(), feed_dict=initializer_feed_dict)
		for i in random.sample(range(N),20000):
			_,mag = self.sess.run([it,test_err_mag], feed_dict={focus_inpt: samples[i,:]})
			print(str(i) + " " + str(mag))

		return self.sess.run(ret.A)

	def test_local_errors(self,inpt,human_labels):
		return self.sess.run(self.test_err, feed_dict={self.test_inpt:inpt, self.test_human_labels: human_labels})

	def get_filename(self):
		return os.path.splitext(os.path.basename(__file__))[0]

TRAIN = MultiDataset(
		[
			os.path.expanduser("~/mydatasets/3_3_1/ds/"),
		],
		{
			"machine_labels": "mean_agg_tr.h5",
			"human_labels": "proofread.h5",
			"samples": "samples.h5",
		}
)
args = {
	"devices": get_device_list(),
	"patch_size": tuple(discrim_net.patch_size_suggestions([2,3,3])[0]),
	"name": "test",
}

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)
with tf.device(args["devices"][0]):
	main_model = DiscrimModel(**args)
#main_model.restore(zenity_workaround())
print("model initialized")
if __name__ == '__main__':
	dataset.h5write(os.path.join(TRAIN.directories[0], "errors.h5"), np.squeeze(main_model.inference(TRAIN.machine_labels[0], TRAIN.human_labels[0], TRAIN.samples[0]),axis=(0,4)))
