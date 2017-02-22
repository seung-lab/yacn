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
from dataset import (gen, Dataset2, alternating_iterator)
from experiments import save_experiment, repo_root


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
			_,maxlabel = np.shape(valid)
			#all of these should be five dimensional!
			full_image = Volume(scv(full_image), self.padded_patch_size)
			full_human_labels = Volume(scv(full_human_labels), self.padded_patch_size)
			full_machine_labels = Volume(scv(full_machine_labels), self.padded_patch_size)
			valid = scv(valid)


		with tf.name_scope('params'):
			self.step = tf.Variable(0)
			forward = basic_net2.make_forward_net(patch_size, 2, nvec_labels)

		params_var_list = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope='params')

		self.iteration_type=tf.placeholder(tf.int64, shape=())
		self.image_feed =tf.placeholder(tf.float32, shape=self.padded_patch_size)
		self.mask_feed = tf.placeholder(tf.float32, shape=self.padded_patch_size)
		self.default_train_dict={self.iteration_type:0}

		iteration_type = self.iteration_type
		train_samples=samples
		test_samples=samples
		#focus=tf.reshape(random_row(tf.cond(tf.equal(self.iteration_type,0),lambda: tf.identity(train_samples),lambda: tf.identity(test_samples))),(3,))
		focus = np.array([0,50,500,500,0])

		rr = RandomRotationPadded()
		image = rr(full_image[focus])
		human_labels = rr(full_human_labels[focus])



		central_label_set = tf.scatter_nd(tf.reshape(extract_central(human_labels),[1,1]), [1], [maxlabel])

		
		#0 means that this label is removed
		#ensure that the central object is not masked, and also ensure that only valid objects are masked.
		masked_label_set = tf.maximum(tf.maximum(tf.to_int32(rand_bool([maxlabel],0.25)), central_label_set), 1-valid[0,:])
		mask = tf.to_float(tf.gather(masked_label_set, human_labels))
		central = tf.to_float(tf.gather(central_label_set, human_labels))

		"""
		central_index=human_labels[0,patch_size[0]/2,patch_size[1]/2,patch_size[2]/2,0]
		remap = tf.where(tf.logical_or(rand_bool([10000], 0.25),tf.equal(np.array(range(10000),dtype=np.int32),central_index)), np.array(range(10000)), np.zeros(10000))
		human_labels = tf.gather(remap, human_labels)
		human_labels = unique(human_labels)
		mask = tf.to_float(tf.minimum(human_labels, 1))
		"""

		vector_labels = forward(tf.concat([image,mask],4))
		vector_labels_test = forward(tf.concat([self.image_feed, self.mask_feed],4))

		with tf.name_scope("loss"):
			"""
			sub_human_labels = human_labels[:,
					int(0.45*patch_size[0]):int(0.55*patch_size[0]),
					int(0.45*patch_size[1]):int(0.55*patch_size[1]),
					int(0.45*patch_size[2]):int(0.55*patch_size[2]),:]
			central_labels = unique_list(sub_human_labels,maxn)
			central_labels_mask = tf.to_float(tf.gather(central_labels, human_labels))
			"""

			loss1=0
			loss2=0
			loss3=0
			#loss1, prediction = label_loss_fun(vector_labels, human_labels, central_labels, central)
			#loss2, long_range_affinities = long_range_loss_fun(vector_labels, human_labels, offsets, mask)
			#loss = loss1 + loss2
			guess = affinity(extract_central(vector_labels),vector_labels)
			truth = label_diff(human_labels, extract_central(human_labels))
			loss3 = tf.reduce_sum(bounded_cross_entropy(guess,truth))
			loss=loss1 + loss2 + loss3

		def training_iteration():
			optimizer = tf.train.AdamOptimizer(0.0003, epsilon=0.1)
			train_op = optimizer.minimize(loss)
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
								image_summary("human_labels", tf.to_float(human_labels)),
							   image_summary("vector_labels", vector_labels),
							   image_summary("expansion", affinity(vector_labels,vector_labels[:,patch_size[0]/2:patch_size[0]/2+1, patch_size[1]/2:patch_size[1]/2+1, patch_size[2]/2:patch_size[2]/2+1,:]))
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
		return self.sess.run(self.vector_labels2, feed_dict={iteration_type: 0, self.image_feed: image, self.mask_feed: mask})



	def train_feed_dict(self):
		return self.default_train_dict

def length_scale(x):
	if x == 0:
		return -1
	else:
		return math.log(abs(x))


def valid_pair(x, y, strict=False):
	return x == 0 or y == 0 or (
		(not strict or length_scale(x) >= length_scale(y)) and abs(
			length_scale(x) - length_scale(y)) <= math.log(3.1))


def valid_offset(x):
	return x > (0,0,0) and \
	valid_pair(4 * x[0], x[1], strict=True) and \
	valid_pair(4 * x[0], x[2], strict=True) and \
	valid_pair(x[1],x[2])

volpath = os.path.expanduser("/usr/people/jzung/seungmount/Omni/TracerTasks/pinky/proofreading/chunk_16513-18560_28801-30848_4003-4258.omni.files/")
TRAIN = Dataset2(volpath, {"image": "image.h5", "human_labels":"proofread.h5", "machine_labels":"mean0.3.h5", "valid": "valid.h5"})

X,Y,Z=TRAIN.image.shape
print(TRAIN.valid.shape)
def subarray(x):
	return x[:,0:X/3,0:Y,0:Z,:]
args = {
	"offsets": filter(valid_offset, itertools.product(
		[-3, -1, 0, 1, 3],
		[-27, -9, -3, -1, 0, 1, 3, 9, 27],
		[-27, -9, -3, -1, 0, 1, 3, 9, 27])),
	"devices": ["/gpu:0"],
	"patch_size": (32, 318, 318),
	"nvec_labels": 6,
	"maxn": 40,

	#dtype=float32, shape=(n,z,y,x,1)
	"full_image": subarray(np.reshape(TRAIN.image,(1,X,Y,Z,1))),

	#dtype=int32, shape=(n,z,y,x,1)
	"full_human_labels": subarray(np.reshape(TRAIN.human_labels,(1,X,Y,Z,1))),

	#dtype=int32, shape=(n,z,y,x,1)
	"full_machine_labels": subarray(np.reshape(TRAIN.machine_labels,(1,X,Y,Z,1))),

	"valid": np.reshape(TRAIN.valid.astype(np.int32), (1,-1)),

	"samples": np.concatenate([np.random.randint(low=x/2, high=y-x/2, size=(100000,1),dtype=np.int32) for x,y in zip((32,158,158), TRAIN.image.shape)],axis=1),
	"name": "test",
}

#New setup
#A sample is a list of supervoxel ids, a window position, and a volume id
#We want to be able to train with multiple volumes.

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)
with tf.device(args["devices"][0]):
	main_model = VectorLabelModel(**args)

print("initialized")
TRAIN=None
args=None
gc.collect()
#import pythonzenity
#if pythonzenity.Question(text="Restore from checkpoint?") == -8:
#	main_model.restore(pythonzenity.FileSelection())

if __name__ == '__main__':
	main_model.train(nsteps=1000000)
	print("done")
