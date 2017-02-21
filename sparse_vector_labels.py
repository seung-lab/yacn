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

import tensorflow as tf
from tensorflow.python.client import timeline

from utils import *
from dataset import (gen, Dataset2, SNEMI3D_TRAIN_DIR, SNEMI3D_TEST_DIR,
					 alternating_iterator)
from experiments import save_experiment, repo_root


class VectorLabelModel(Model):

	def __init__(self, patch_size, offsets, full_image, full_human_labels, full_machine_labels, samples,
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
		self.propagator = identity_matrix(nvec_labels)
		# tf.matrix_inverse(self.propagator)
		self.inv_propagator = identity_matrix(nvec_labels)

		config = tf.ConfigProto(
			gpu_options = tf.GPUOptions(allow_growth=True),
			allow_soft_placement=True,
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
			full_image = Volume(scv(full_image), self.padded_patch_size)
			full_human_labels = Volume(scv(full_human_labels), self.padded_patch_size)
			full_machine_labels = Volume(scv(full_machine_labels), self.padded_patch_size)

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

		central_index=human_labels[0,patch_size[0]/2,patch_size[1]/2,patch_size[2]/2,0]

		remap = tf.where(tf.logical_or(rand_bool([10000], 0.25),tf.equal(np.array(range(10000),dtype=np.int32),central_index)), np.array(range(10000)), np.zeros(10000))
		human_labels = tf.gather(remap, human_labels)

		human_labels = unique(human_labels)
		mask = tf.to_float(tf.minimum(human_labels, 1))

		vector_labels = forward(tf.concat([image,mask],4))
		vector_labels_test = forward(tf.concat([self.image_feed, self.mask_feed],4))

		with tf.name_scope("loss"):
			sub_human_labels = human_labels[:,
					int(0.45*patch_size[0]):int(0.55*patch_size[0]),
					int(0.45*patch_size[1]):int(0.55*patch_size[1]),
					int(0.45*patch_size[2]):int(0.55*patch_size[2]),:]
			central_labels = unique_list(sub_human_labels,maxn)
			central_labels_mask = tf.to_float(tf.gather(central_labels, human_labels))

			loss1=0
			loss2=0
			loss1, prediction = self.label_loss_fun(
				vector_labels, human_labels, central_labels, central_labels_mask)
			loss2, long_range_affinities = self.long_range_loss_fun(
				vector_labels, human_labels, offsets, central_labels, central_labels_mask)
			#loss = loss1 + loss2
			loss=loss1 + loss2

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
							   image_summary("central_labels_mask", tf.to_float(central_labels_mask)),
							   image_summary("expansion", self.affinity(vector_labels,vector_labels[:,patch_size[0]/2:patch_size[0]/2+1, patch_size[1]/2:patch_size[1]/2+1, patch_size[2]/2:patch_size[2]/2+1,:]))
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

	def affinity(self, x, y):
		n=len(static_shape(x))
		displacement = x - y
		interaction = tf.reduce_sum(
			displacement * displacement,
			reduction_indices=[n-1],
			keep_dims=True)
		return tf.exp(-0.5 * interaction)

	def long_range_loss_fun(self, vec_labels, human_labels, offsets, central_labels, central_labels_mask):
		cost = 0
		otpts = {}


		for i, offset in enumerate(offsets):
			guess = self.affinity(
				*get_pair(vec_labels,offset,self.patch_size))
			truth = self.label_diff(
					*get_pair(human_labels,offset,self.patch_size))

			mask1, mask2 = get_pair(central_labels_mask, offset, self.patch_size)

			otpts[offset] = guess

			cost += tf.reduce_sum(tf.maximum(mask1, mask2) *
								  bounded_cross_entropy(guess, truth))

		return cost, otpts

	def label_diff(self, x, y):
		return tf.to_float(tf.equal(x, y))

	def test(self, image, mask):
		return self.sess.run(self.vector_labels2, feed_dict={iteration_type: 0, self.image_feed: image, self.mask_feed: mask})

	def batch_interaction(self, mu0, cov0, mu1, cov1):
		nvec_labels = self.nvec_labels
		maxn = self.maxn
		A = tf.reshape(self.propagator, [1, 1, nvec_labels, nvec_labels])
		invA = tf.reshape(self.inv_propagator, [1, 1, nvec_labels, nvec_labels])
		mu0 = tf.reshape(mu0, [maxn, 1, nvec_labels, 1])
		mu1 = tf.reshape(mu1, [1, maxn, nvec_labels, 1])
		cov0 = tf.reshape(cov0, [maxn, 1, nvec_labels, nvec_labels])
		cov1 = tf.reshape(cov1, [1, maxn, nvec_labels, nvec_labels])
		identity = tf.reshape(tf.diag(tf.ones((nvec_labels,))), [
							  1, 1, nvec_labels, nvec_labels])
		identity2 = tf.reshape(identity_matrix(
			2 * nvec_labels), [1, 1, 2 * nvec_labels, 2 * nvec_labels])

		cov0 = cov0 + 0.0001 * identity
		cov1 = cov1 + 0.0001 * identity

		delta = mu1 - mu0
		delta2 = vec_cat(delta, delta)

		sqcov0 = tf.tile(tf.cholesky(cov0), [1, maxn, 1, 1])
		sqcov1 = tf.tile(tf.cholesky(cov1), [maxn, 1, 1, 1])
		invA = tf.tile(invA, [maxn, maxn, 1, 1])
		A = tf.tile(A, [maxn, maxn, 1, 1])

		scale = block_cat(
			sqcov0,
			tf.zeros_like(sqcov0),
			tf.zeros_like(sqcov0),
			sqcov1)
		M = batch_matmul(
			batch_transpose(scale),
			block_cat(
				invA,
				invA,
				invA,
				invA),
			scale) + identity2
		scale2 = block_cat(invA, tf.zeros_like(A), tf.zeros_like(A), invA)

		v = batch_matmul(
			tf.matrix_inverse(M),
			batch_transpose(scale),
			scale2,
			delta2)

		ret1 = 1 / tf.sqrt(tf.matrix_determinant(M))
		ret2 = tf.exp(-0.5 * (batch_matmul(batch_transpose(delta),
					  invA, delta) - batch_matmul(batch_transpose(v), M, v)))

		ret = ret1 * tf.squeeze(ret2, [2, 3])

		return ret

	def label_loss_fun(self, vec_labels, human_labels, central_labels, central_labels_mask):
		maxn = self.maxn
		weight_mask = tf.maximum(identity_matrix(maxn),tf.maximum(tf.reshape(central_labels,[-1,1])*tf.ones([1,maxn]),tf.ones([maxn,1])*tf.reshape(central_labels,[1,-1])))


		human_labels = tf.reshape(human_labels, [-1])
		vec_labels = tf.reshape(vec_labels, [-1, self.nvec_labels])
		central_labels_mask = tf.reshape(central_labels_mask, [-1])
		sums = tf.unsorted_segment_sum(vec_labels, human_labels, maxn)
		weights = tf.to_float(tf.unsorted_segment_sum(tf.ones(static_shape(human_labels)), human_labels, maxn))
		safe_weights = tf.maximum(weights, 0.1)
		
		means = sums / tf.reshape(safe_weights, [maxn,1])
		centred_vec_labels = vec_labels - tf.gather(means, human_labels)
		full_covs = tf.reshape(centred_vec_labels, [-1, self.nvec_labels, 1]) * tf.reshape(centred_vec_labels, [-1, 1, self.nvec_labels])
		sqsums = tf.unsorted_segment_sum(full_covs, human_labels, maxn)
		
		pack_means = means
		pack_covs = sqsums / tf.reshape(safe_weights, [maxn,1,1])
		pack_weights = weights

		predictions = self.batch_interaction(
			pack_means, pack_covs, pack_means, pack_covs)
		predictions = 0.99998 * predictions + 0.00001
		objective = identity_matrix(maxn)
		weight_matrix = weight_mask * tf.reshape(
			tf.sqrt(pack_weights), [-1, 1]) * tf.reshape(
			tf.sqrt(pack_weights), [1, -1])

		cost = - objective * tf.log(predictions) - \
			(1 - objective) * tf.log(1 - predictions)

		return tf.reduce_sum(weight_matrix * cost) + tf.reduce_sum(tf.reshape(central_labels_mask,[-1,1]) * bounded_cross_entropy(self.affinity(vec_labels, centred_vec_labels),1)), predictions


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

TRAIN = Dataset2(SNEMI3D_TRAIN_DIR, {"image": "image.h5", "human_labels":"human_labels.h5", "machine_labels":"machine_labels.h5"})
args = {
	"offsets": filter(valid_offset, itertools.product(
		[-3, -1, 0, 1, 3],
		[-27, -9, -3, -1, 0, 1, 3, 9, 27],
		[-27, -9, -3, -1, 0, 1, 3, 9, 27])),
	"devices": ["/gpu:0"],
	"patch_size": (32, 158, 158),
	"nvec_labels": 6,
	"maxn": 40,

	#dtype=float32, shape=(n,z,y,x,1)
	"full_image": np.reshape(TRAIN.image,(1,100,1024,1024,1)),

	#dtype=int32, shape=(n,z,y,x,1)
	"full_human_labels": np.reshape(TRAIN.human_labels,(1,100,1024,1024,1)),

	#dtype=int32, shape=(n,z,y,x,1)
	"full_machine_labels": np.reshape(TRAIN.machine_labels,(1,100,1024,1024,1)),


	"samples": np.concatenate([np.random.randint(low=x/2, high=y-x/2, size=(100000,1),dtype=np.int32) for x,y in zip((32,158,158), TRAIN.image.shape)],axis=1),
	"name": "test"
}

#New setup
#A sample is a list of supervoxel ids, a window position, and a volume id
#We want to be able to train with multiple volumes.

import pythonzenity
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)
with tf.device(args["devices"][0]):
	main_model = VectorLabelModel(**args)

#if pythonzenity.Question(text="Restore from checkpoint?") == -8:
#	main_model.restore(pythonzenity.FileSelection())

if __name__ == '__main__':
	main_model.train(nsteps=1000000)
	print("done")
