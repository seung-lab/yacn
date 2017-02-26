#Let's change the loss function to only care about reconstructing the objects which pass through the centre of the volume. It's okay to reconstruct objects that don't touch the centre, but we need to be sure that we reconstruct those that are in the centre. In fact, we'll usually only care about the reconstruction of a particular object near the centre.

from __future__ import print_function
import numpy as np
import os
from datetime import datetime
import time
import math
import itertools
import threading
import pprint
from convkernels3d import *
from activations import *
import basic_net2
import gc

import tensorflow as tf
from tensorflow.python.client import timeline

from utils import *
from loss_functions import *
import dataset


class VectorLabelModel(Model):

	def __init__(self, patch_size, offsets, full_image, full_human_labels, full_machine_labels, valid, samples,
				 nvec_labels, maxn,
				 devices, name=None):

		self.name=name
		self.summaries = []
		self.devices = devices
		self.patch_size = patch_size
		self.padded_patch_size = (1,) + patch_size + (1,)
		self.offsets = offsets
		self.maxn = maxn
		self.nvec_labels = nvec_labels

		config = tf.ConfigProto(
			gpu_options = tf.GPUOptions(allow_growth=True),
			allow_soft_placement=True,
			#log_device_placement=True,
			#gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
		)
		config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
		self.sess = tf.Session(config=config)
		self.run_metadata = tf.RunMetadata()

		
		initializer_feed_dict={}
		def scv(x):
			return static_constant_variable(x,initializer_feed_dict)

		with tf.device("/cpu:0"):
			#all of these should be five dimensional!
			full_image = MultiVolume(map(scv,full_image), self.padded_patch_size)
			full_human_labels = MultiVolume(map(scv,full_human_labels), self.padded_patch_size)
			full_machine_labels = MultiVolume(map(scv,full_machine_labels), self.padded_patch_size)
			samples = MultiVolume(map(scv,samples), (1,3), indexing='CORNER')
			valid = map(scv,valid)

		with tf.name_scope('params'):
			self.step = tf.Variable(0)
			forward = basic_net2.make_forward_net(patch_size, 2, nvec_labels)

		params_var_list = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope='params')

		self.iteration_type=tf.placeholder(tf.int64, shape=())
		self.image_feed =tf.placeholder(tf.float32, shape=self.padded_patch_size)
		self.mask_feed = tf.placeholder(tf.float32, shape=self.padded_patch_size)
		self.default_train_dict={self.iteration_type:0}
		vector_labels_test = forward(tf.concat([self.image_feed, self.mask_feed],4))
		self.vector_labels_test = vector_labels_test
		self.expansion_test = affinity(extract_central(vector_labels_test), vector_labels_test)

		iteration_type = self.iteration_type
		train_samples=samples
		test_samples=samples

		loss = 0
		for i,d in enumerate(devices):
			with tf.device(d):
				with tf.name_scope("device"+str(i)):
					vol_id=0
					maxlabel = tf.shape(valid[vol_id])[0]
					focus=tf.concat([[0],tf.reshape(samples[vol_id,('RAND',0)],(3,)),[0]],0)

					rr = RandomRotationPadded()
					image = rr(full_image[vol_id,focus])
					human_labels = rr(full_human_labels[vol_id,focus])

					central_label = extract_central(human_labels)
					is_valid = tf.to_float(valid[vol_id][tf.reshape(central_label,[])])
					central_label_set = tf.scatter_nd(tf.reshape(central_label,[1,1]), [1], [maxlabel])

					
					#0 means that this label is removed
					#ensure that the central object is not masked, and also ensure that only valid objects are masked.
					masked_label_set = tf.maximum(tf.maximum(tf.to_int32(rand_bool([maxlabel],0.25)), central_label_set), 1-valid[vol_id])

					#ensure that zero is masked out
					#masked_label_set = tf.minimum(masked_label_set, tf.concat(tf.zeros((1,),dtype=tf.int32),tf.ones((maxlabel-1,),dtype=tf.int32)))
					masked_label_set = tf.concat([tf.zeros([1],dtype=tf.int32),masked_label_set[1:]],0)
					mask = tf.to_float(tf.gather(masked_label_set, human_labels))
					central = tf.to_float(tf.gather(central_label_set, human_labels))

					vector_labels = forward(tf.concat([image,mask],4))
					
					central_vector = tf.reduce_sum(central * vector_labels, reduction_indices = [1,2,3], keep_dims=True)/ tf.reduce_sum(central, keep_dims=False) 

					with tf.name_scope("loss"):
						loss1=0
						loss2=0
						loss3=0
						#loss1, prediction = label_loss_fun(vector_labels, human_labels, central_labels, central)
						#loss2, long_range_affinities = long_range_loss_fun(vector_labels, human_labels, offsets, mask)
						guess = affinity(central_vector,vector_labels)
						truth = label_diff(human_labels, extract_central(human_labels))
						loss3 = tf.reduce_sum(bounded_cross_entropy(guess,truth)) * is_valid
						loss += loss1 + loss2 + loss3

		def training_iteration():
			optimizer = tf.train.AdamOptimizer(0.002, epsilon=0.1)
			train_op = optimizer.minimize(loss, colocate_gradients_with_ops=True)
			with tf.control_dependencies([train_op]):
				train_op = tf.group(self.step.assign_add(1), tf.Print(
					0, [self.step, iteration_type, loss],
					message="step|iteration_type|loss"))
				quick_summary_op = tf.summary.merge([
					tf.summary.scalar("loss_train", loss),
				])
			return train_op, quick_summary_op

		def test_iteration():
			quick_summary_op = tf.summary.merge(
				[tf.summary.scalar("loss_test", loss)])
			return tf.no_op(), quick_summary_op

		train_op, quick_summary_op = tf.cond(
			tf.equal(self.iteration_type, 0),
			training_iteration, test_iteration)

		#self.summaries.extend(
		#	[image_slice_summary(
		#		"boundary_{}".format(key), long_range_affinities[key])
		#		for key in long_range_affinities])
		self.summaries.extend([image_summary("image", image),
								image_summary("mask", tf.to_float(mask)),
								#image_summary("human_labels", tf.to_float(human_labels)),
							   image_summary("vector_labels", vector_labels),
							   image_summary("guess", guess),
							   image_summary("truth", truth),
							   ])
		#self.summaries.extend([tf.summary.image("prediction", tf.reshape(prediction,[1,maxn,maxn,1]))])
		summary_op = tf.summary.merge(self.summaries)

		init = tf.global_variables_initializer()
		self.sess.run(init, feed_dict=initializer_feed_dict)

		self.saver = tf.train.Saver(var_list=params_var_list)
		self.train_op = train_op
		self.quick_summary_op = quick_summary_op

		self.summary_op = summary_op
	
	def get_filename(self):
		return os.path.splitext(os.path.basename(__file__))[0]

	def test(self, image, mask):
		image = dataset.prep("image",image)
		mask = dataset.prep("image",mask)
		return self.sess.run(self.expansion_test, feed_dict={self.iteration_type: 1, self.image_feed: image, self.mask_feed: mask})

	def train_feed_dict(self):
		return self.default_train_dict

if __name__ == '__main__':
	TRAIN = dataset.MultiDataset(
			[
				os.path.expanduser("/usr/people/jzung/seungmount/Omni/TracerTasks/pinky/proofreading/chunk_16513-18560_28801-30848_4003-4258.omni.files/"),
				#os.path.expanduser("/usr/people/jzung/seungmount/Omni/TracerTasks/pinky/proofreading/chunk_18049-20096_30337-32384_4003-4258.omni.files/ds/"),
			],
			{
				"image": "image.h5",
				"human_labels": "proofread.h5",
				"machine_labels": "mean0.3.h5",
				"valid": "valid.h5",
			}
	)
else:
	TRAIN = dataset.MultiDataset(
			["/usr/people/jzung/seungmount/research/datasets/dummy/"], {"image": "image.h5", "human_labels": "human_labels.h5", "machine_labels": "machine_labels.h5", "valid": "valid.h5"})

patch_size=(33,318,318)
args = {
	"offsets": filter(valid_offset, itertools.product(
		[-3, -1, 0, 1, 3],
		[-27, -9, 0, 9, 27],
		[-27, -9, 0, 9, 27])),
	"devices": get_device_list(),
	"patch_size": patch_size,
	"nvec_labels": 6,
	"maxn": 40,

	"full_image": TRAIN.image,

	"full_human_labels": TRAIN.human_labels,

	"full_machine_labels": TRAIN.machine_labels,

	"valid": TRAIN.valid,

	"samples": [np.concatenate([np.random.randint(low=x/2, high=y-x/2, size=(100000,1),dtype=np.int32) for x,y in zip(patch_size, [256,2000,2000])],axis=1)],
	"name": "test",
}

#New setup
#A sample is a list of supervoxel ids, a window position, and a volume id
#We want to be able to train with multiple volumes.

#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(args)
main_model = VectorLabelModel(**args)

print("initialized")
TRAIN=None
args=None
gc.collect()

if __name__ == '__main__':
	main_model.train(nsteps=1000000, checkpoint_interval=200)
	print("done")
