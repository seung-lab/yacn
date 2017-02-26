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

import tensorflow as tf
from tensorflow.python.client import timeline

from utils import *
from dataset import MultiDataset
import pythonzenity

class DiscrimModel(Model):
	def __init__(self, patch_size, 
				 truth_data,
				 lies_data,
				 samples,
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

		initializer_feed_dict={}
		def scv(x):
			return static_constant_variable(x,initializer_feed_dict)

		with tf.device("/cpu:0"):
			vol_id=0
			full_labels_truth = MultiVolume(map(scv,truth_data), self.padded_patch_size)
			full_labels_lies = MultiVolume(map(scv,lies_data), self.padded_patch_size)
			samples = MultiVolume(map(scv,samples), (1,3), indexing='CORNER')
	
		with tf.name_scope('params'):
			self.step=tf.Variable(0)
			discrim, reconstruct = discrim_net.make_forward_net(patch_size,1,1)

		loss=0
		reconstruction_loss=0
		for i,d in enumerate(devices):
			with tf.device(d):
				with tf.name_scope("gpu"+str(i)):
					focus_lies=tf.concat([[0],tf.reshape(samples[vol_id,('RAND',0)],(3,)),[0]],0)
			
					rr=RandomRotationPadded()

					#1 is correct and 0 is incorrect
					#truth_glimpse = rr(equal_to_centre(full_labels_truth[vol_id,focus_truth]))
					lies_glimpse = rr(equal_to_centre(full_labels_lies[vol_id,focus_lies]))
					lies_compare = rr(equal_to_centre(full_labels_truth[vol_id,focus_lies]))
					human_labels = rr(full_labels_truth[vol_id,focus_lies])

					#truth_discrim, truth_discrim_mid = discrim(truth_glimpse)
					lies_discrim, lies_discrim_mid = discrim(lies_glimpse)

					with tf.device("/cpu:0"):
						errors = localized_errors(lies_glimpse, human_labels)

					print(lies_discrim)
					print(lies_discrim_mid)

					loss += 0.1 * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = lies_discrim_mid, labels=errors))
					loss += tf.nn.sigmoid_cross_entropy_with_logits(logits=lies_discrim, labels=has_error(lies_glimpse, human_labels))

					reconstruction = reconstruct(lies_glimpse)

					reconstruction_loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=lies_compare))

		self.test_inpt = tf.placeholder(shape=self.padded_patch_size, dtype=tf.float32)
		self.test_otpt = discrim(self.test_inpt)

		var_list = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope='params')

		optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.9, epsilon=0.1)
		train_op = optimizer.minimize(1e5*loss + reconstruction_loss, colocate_gradients_with_ops=True, var_list = var_list)

		with tf.control_dependencies([train_op]):
			with tf.control_dependencies([self.step.assign_add(1)]):
				train_op = tf.group(
						tf.Print(0, [tf.identity(self.step), loss], message="step|loss"),
						)
						
				quick_summary_op = tf.summary.merge([
					tf.summary.scalar("loss", loss),
					tf.summary.scalar("reconstruction_loss", reconstruction_loss),
					#tf.summary.scalar("truth_discrim", truth_discrim),
					tf.summary.scalar("lies_discrim", lies_discrim),
				])

		self.summaries.append(tf.summary.scalar("loss",loss))
		summary_op = tf.summary.merge(self.summaries)

		init = tf.global_variables_initializer()
		self.sess.run(init, feed_dict=initializer_feed_dict)

		self.saver = tf.train.Saver(var_list=var_list)
		self.train_op = train_op
		self.quick_summary_op = quick_summary_op
		self.summary_op = summary_op


	def test(self,x):
		x=[self.sess.run(self.test_otpt, feed_dict={self.test_inpt:x}) for i in xrange(4)]
		return sum(x)/4

	def get_filename(self):
		return os.path.splitext(os.path.basename(__file__))[0]

	def interrupt(self):
		return
	def train_feed_dict(self):
		return {}

TRAIN = MultiDataset(
		[
			os.path.expanduser("/usr/people/jzung/seungmount/Omni/TracerTasks/pinky/proofreading/chunk_16513-18560_28801-30848_4003-4258.omni.files/ds/"),
			#os.path.expanduser("/usr/people/jzung/seungmount/Omni/TracerTasks/pinky/proofreading/chunk_18049-20096_30337-32384_4003-4258.omni.files/ds/"),
		],
		{
			"machine_labels": "mean_labels.h5",
			"human_labels": "human_labels.h5",
			"samples": "mean_samples.h5",
		}
)
		
args = {
	"devices": ["/gpu:0"],
	"patch_size": (33, 158, 158),
	"name": "test",
	"truth_data": TRAIN.human_labels,
	"lies_data": TRAIN.machine_labels,
	"samples": TRAIN.samples,
}

#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(args)
with tf.device(args["devices"][0]):
	main_model = DiscrimModel(**args)
print("model initialized")
if __name__ == '__main__':
	main_model.train(nsteps=1000000)
