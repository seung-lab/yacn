# plan: We need to give the autoencoder some more memory,
# and give it access to the higher level features already computed.

# implement hard attention

# We should have local feature extraction and recurrent tracing
# You should think of the model as being inactive most of the time.
# We need to implement hard attention in order to move to the next window
# Memory is is spatially indexed

#We need an adversary which checks the output to see if it is distinguishable from the output on test data.


from __future__ import print_function
import numpy as np
import os
from datetime import datetime
import time
import math
import itertools
import threading
import pprint
from convkernels import *
from activations import *
import basic_net

import tensorflow as tf
from tensorflow.python.client import timeline

from utils import *
from dataset import (gen, Dataset2, SNEMI3D_TRAIN_DIR, SNEMI3D_TEST_DIR,
					 alternating_iterator)
from experiments import save_experiment, repo_root


class VectorLabelModel(Model):

	def __init__(self, patch_size, offsets, mask_boundary, full_image, full_human_labels, samples,
				 nvec_labels, maxn,
				 devices, name=None):

		self.name=name
		self.summaries = []
		self.devices = devices
		self.patch_size = patch_size
		self.offsets = offsets
		self.mask_boundary = mask_boundary
		self.maxn = maxn
		self.nvec_labels = nvec_labels
		self.propagator = identity_matrix(nvec_labels)
		# tf.matrix_inverse(self.propagator)
		self.inv_propagator = identity_matrix(nvec_labels)

		config = tf.ConfigProto(
			# gpu_options = tf.GPUOptions(allow_growth=True),
			allow_soft_placement=True,
			#gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
		)
		self.sess = tf.Session(config=config)
		self.run_metadata = tf.RunMetadata()

		
		initializer_feed_dict={}
		def scv(x):
			return static_constant_variable(x,initializer_feed_dict)

		with tf.device("/cpu:0"):
			full_image = Volume(scv(full_image), patch_size)
			full_human_labels = Volume(scv(full_human_labels), patch_size)

		with tf.name_scope('params'):
			self.step = tf.Variable(0)
			forward = basic_net.make_forward_net(patch_size, 2, nvec_labels)

		var_list = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope='params')

		self.iteration_type=0
		iteration_type = self.iteration_type
		train_samples=samples
		test_samples=samples
		focus=tf.reshape(random_row(tf.cond(tf.equal(self.iteration_type,0),lambda: tf.identity(train_samples),lambda: tf.identity(test_samples))),(3,))

		image = full_image[focus]
		human_labels = full_human_labels[focus]
		remap = tf.select(rand_bool([10000], 0.33), np.array(range(10000)), np.zeros(10000))
		human_labels = tf.to_int32(tf.gather(tf.to_float(remap), human_labels)) #gather is only implemented on gpu for floats
		tmp = unique(human_labels) + 1
		human_labels = tmp * tf.to_int32(tf.not_equal(human_labels, 0))
		mask = tf.to_float(tf.minimum(human_labels, 1))

		vector_labels = forward(tf.concat(3,[tf_pad_shape(image),tf_pad_shape(mask)]))

		with tf.name_scope("loss"):
			loss1, prediction = self.label_loss_fun(
				vector_labels, human_labels)
			loss2, long_range_affinities = self.long_range_loss_fun(
				vector_labels, human_labels, offsets)
			loss = loss1 + loss2

		def training_iteration():
			optimizer = tf.train.AdamOptimizer(0.0001, epsilon=0.1)
			train_op = optimizer.minimize(loss)
			with tf.control_dependencies([train_op]):
				train_op = tf.group(self.step.assign_add(1), tf.Print(
					0, [self.step, iteration_type, loss],
					message="step|iteration_type|loss"))
				quick_summary_op = tf.merge_summary([
					tf.scalar_summary("loss_train", loss),
				])
			return train_op, quick_summary_op

		def test_iteration():
			quick_summary_op = tf.merge_summary(
				[tf.scalar_summary("loss_test", loss)])
			return tf.no_op(), quick_summary_op

		train_op, quick_summary_op = tf.cond(
			tf.equal(self.iteration_type, 0),
			training_iteration, test_iteration)

		#self.summaries.extend(
		#	[image_slice_summary(
		#		"boundary_{}".format(key), long_range_affinities[key])
		#		for key in long_range_affinities])
		self.summaries.extend([image_summary("image", tf_pad_shape(image)),
								image_summary("mask", tf_pad_shape(tf.to_float(mask))),
								image_summary("human_labels", tf_pad_shape(tf.to_float(human_labels))),
							   image_summary("vector_labels", vector_labels)
							   ])
		self.summaries.extend([tf.image_summary("prediction",
							tf.reshape(prediction,[1,maxn,maxn,1]))])
		summary_op = tf.merge_summary(self.summaries)

		init = tf.initialize_all_variables()
		self.sess.run(init, feed_dict=initializer_feed_dict)

		self.saver = tf.train.Saver()
		self.train_op = train_op
		self.quick_summary_op = quick_summary_op

		self.summary_op = summary_op
	
	def init_log(self):
		print ('What do you want to call this experiment?')
		date = datetime.now().strftime("%j-%H-%M-%S")
		filename=os.path.splitext(os.path.basename(__file__))[0]
		if self.name is None:
			self.name=raw_input('run name: ')
		exp_name = date + '-' + self.name

		logdir = os.path.expanduser("~/experiments/{}/{}/".format(filename,exp_name))
		self.logdir = logdir

		print('logging to {}, also created a branch called {}'
			  .format(logdir, exp_name))
		if not os.path.exists(logdir):
			os.makedirs(logdir)
		#save_experiment(exp_name)
		self.summary_writer = tf.train.SummaryWriter(
			logdir, graph=self.sess.graph)

	def restore(self, modelpath):
		modelpath = os.path.expanduser(modelpath)
		self.saver.restore(self.sess, modelpath)

	def affinity(self, x, y):
		displacement = x - y
		interaction = tf.reduce_sum(
			displacement * displacement,
			reduction_indices=[3],
			keep_dims=True)
		return tf.exp(-0.5 * interaction)

	def selection_to_attention(self, vector_labels, selection):
		return tf.reduce_sum(selection * vector_labels,
							 reduction_indices=[0, 1, 2],
							 keep_dims=True) / tf.reduce_sum(selection)

	def attention_to_selection(self, vector_labels, attention):
		return self.affinity(vector_labels, attention)

	def long_range_loss_fun(self, vec_labels, human_labels, offsets):
		human_labels = tf_pad_shape(human_labels)
		cost = 0
		otpts = {}

		if self.mask_boundary:
			mask = tf.to_float(tf.not_equal(self.human_labels, 0))
		else:
			mask = 1

		"""
		if self.mask_unknown:
			mask = mask * \
				tf.to_float(tf.not_equal(self.human_labels, self.maxn))
		else:
			mask = mask * 1
		"""

		for i, offset in enumerate(offsets):
			guess = self.affinity(
				*
				get_pair(
					vec_labels,
					offset,
					self.patch_size))
			truth = self.label_diff(
				*
				get_pair(
					human_labels,
					offset,
					self.patch_size))

			if mask != 1:
				mask1, mask2 = get_pair(mask, offset, self.patch_size)
			else:
				mask1, mask2 = 1, 1

			otpts[offset] = guess

			cost += tf.reduce_sum(mask1 * mask2 *
								  bounded_cross_entropy(guess, truth))

		return cost, otpts

	def label_diff(self, x, y):
		return tf.to_float(tf.equal(x, y))

	def batch_interaction(self, mu0, cov0, mu1, cov1):
		nvec_labels = self.nvec_labels
		maxn = self.maxn
		A = tf.reshape(self.propagator, [1, 1, nvec_labels, nvec_labels])
		invA = tf.reshape(
			self.inv_propagator, [
				1, 1, nvec_labels, nvec_labels])
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

	def label_loss_fun(self, vec_labels, human_labels):
		maxn = self.maxn

		human_labels = tf.reshape(human_labels, [-1])
		vec_labels = tf.reshape(vec_labels, [-1, self.nvec_labels])
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
		weight_matrix = tf.reshape(
			tf.sqrt(pack_weights), [-1, 1]) * tf.reshape(
			tf.sqrt(pack_weights), [1, -1])

		cost = - objective * tf.log(predictions) - \
			(1 - objective) * tf.log(1 - predictions)

		return tf.reduce_sum(weight_matrix * cost), predictions

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
	return x > (
		0,
		0,
		0) and valid_pair(
		4 *
		x[0],
		x[1],
		strict=True) and valid_pair(
			4 *
			x[0],
			x[2],
			strict=True) and valid_pair(
				x[1],
		x[2])

TRAIN = Dataset2(SNEMI3D_TRAIN_DIR, {"image": "image.h5", "human_labels":"human_labels.h5"})
args = {
	"offsets": filter(valid_offset, itertools.product(
		[-3, -1, 0, 1, 3],
		[-27, -9, -3, -1, 0, 1, 3, 9, 27],
		[-27, -9, -3, -1, 0, 1, 3, 9, 27])),
	"devices": ["/gpu:0"],
	"patch_size": (32, 158, 158),
	"mask_boundary": False,
	"nvec_labels": 6,
	"maxn": 40,
	"full_image": TRAIN.image,
	"full_human_labels": TRAIN.human_labels.astype(np.int32),
	"samples": np.concatenate([np.random.randint(low=x/2, high=y-x/2, size=(100000,1),dtype=np.int32) for x,y in zip((32,158,158), TRAIN.image.shape)],axis=1),
	"name": "test"
}
# offsets:
# devices: device used by tensorflow to run the model
# patch_size: size of window input and output by the model
# nvec_labels: size of the vectors of every pixel

import pythonzenity
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)
with tf.device(args["devices"][0]):
	main_model = VectorLabelModel(**args)

if __name__ == '__main__':

	if pythonzenity.Question(text="Restore from checkpoint?") == -8:
		main_model.restore(pythonzenity.FileSelection())

	main_model.train(nsteps=1000000)
	print("done")
