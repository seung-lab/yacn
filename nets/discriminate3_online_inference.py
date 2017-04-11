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
import discrim_net3
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
				 name=None):

		self.name=name
		self.summaries = []
		self.patch_size = patch_size
		self.padded_patch_size = (1,) + patch_size + (1,)

		patchx,patchy,patchz = patch_size

		config = tf.ConfigProto(
			allow_soft_placement=True,
			gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
			#log_device_placement=True,
		)
		self.sess = tf.Session(config=config)
		self.run_metadata = tf.RunMetadata()

		with tf.name_scope('params'):
			self.step=tf.Variable(0)
			discrim, reconstruct = discrim_net3.make_forward_net(patch_size,2,1)
			self.discrim = discrim

		#for some reason we need to initialize here first... Figure this out!
		init = tf.global_variables_initializer()
		self.sess.run(init)

		with tf.name_scope('iteration'):
			self.mask = tf.placeholder(dtype=tf.float32, shape=self.padded_patch_size)
			self.image = tf.placeholder(dtype=tf.float32, shape = self.padded_patch_size)

			discrim_tower = self.discrim(tf.concat([self.mask,self.image],4))
			i=4
			ds_shape = static_shape(discrim_tower[i])
			print(ds_shape)
			expander = compose(*reversed(discrim_net3.range_expanders[0:i]))
			self.otpt = upsample_max(tf.nn.sigmoid(discrim_tower[i]), self.padded_patch_size, expander) * self.mask
			
		var_list = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope='params')

		self.sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='iteration')))

		self.saver = tf.train.Saver(var_list=var_list)
	
	def test(self, image, mask):
		image = dataset.prep("image",image)
		mask = dataset.prep("image",mask)
		ret = self.sess.run(self.otpt, feed_dict={self.image: image, self.mask: mask})
		return ret

	def get_filename(self):
		return os.path.splitext(os.path.basename(__file__))[0]

args = {
	"patch_size": tuple(discrim_net3.patch_size_suggestions([2,3,3])[0]),
	"name": "test",
}
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)
with tf.device("/gpu:0"):
	main_model = DiscrimModel(**args)
print("model initialized")
