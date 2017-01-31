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

import tensorflow as tf
from tensorflow.python.client import timeline

from utils import *
from dataset import (gen, Dataset, SNEMI3D_TRAIN_DIR, SNEMI3D_TEST_DIR,
					 alternating_iterator)
from experiments import save_experiment, repo_root
import pythonzenity


class Model():
	def __init__(self, patch_size, 
				 nvec_labels, maxn,
				 full_image,
				 full_human_labels,
				 devices, name=None):

		self.name=name
		self.summaries = []
		self.devices = devices
		self.patch_size = patch_size
		self.maxn = maxn
		self.nvec_labels = nvec_labels
		self.propagator = identity_matrix(nvec_labels)
		self.inv_propagator = identity_matrix(nvec_labels)

		config = tf.ConfigProto(
			#gpu_options = tf.GPUOptions(allow_growth=True),
			allow_soft_placement=True,
			gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True),
			#log_device_placement=True,
			# intra_op_parallelism_threads=1,
			# inter_op_parallelism_threads=1,
		)
		self.sess = tf.Session(config=config)
		self.run_metadata = tf.RunMetadata()

		with tf.device("/cpu:0"):
			full_trace = tf.Variable(np.zeros_like(full_image),dtype=tf.float32)
			full_image = tf.constant(full_image, dtype=tf.float32)
			full_human_labels = tf.constant(full_human_labels,dtype=tf.int32)

		def get_glimpse(A, focus):
			with tf.device("/cpu:0"):
				corner = focus - np.array([x/2 for x in patch_size],dtype=np.int32)
				corner = tf.Print(corner,[corner])
				corner = tf.unpack(corner)
				return tf.stop_gradient(tf.slice(A,corner,patch_size))
		
		def set_glimpse(A, focus, vol):
			with tf.device("/cpu:0"):
				corner = focus - np.array([x/2 for x in patch_size],dtype=np.int32)
				corner = tf.Print(corner,[corner])
				corner = tf.unpack(corner)
				return tf.stop_gradient(A[corner[0]:corner[0]+patch_size[0],
							corner[1]:corner[1]+patch_size[1],
							corner[2]:corner[2]+patch_size[2]].assign(vol))

		def spawn(proposal_logits, current_focus):
			t,selection_logit=categorical(proposal_logits,selection_logit=True)
			t=tf.pack(t)
			t=tf.Print(t,[t])
			return current_focus + t - np.array([x/2 for x in patch_size],dtype=np.int32), selection_logit 

		with tf.name_scope('params'):
			self.step = tf.Variable(0)
			strides = [(2, 2) for i in xrange(5)]
			sizes = [(4, 4, 1), (4, 4, 2), (4, 4, 4), (4, 4, 8), (4, 4, 16)]

			initial_schemas = [
						FeatureSchema(1,0),
						FeatureSchema(24,1),
						FeatureSchema(28,2),
						FeatureSchema(32,3),
						FeatureSchema(48,4),
						FeatureSchema(64,5)]
			second_schemas = [
						FeatureSchema(nvec_labels,0),
						FeatureSchema(24,1),
						FeatureSchema(28,2),
						FeatureSchema(32,3),
						FeatureSchema(48,4),
						FeatureSchema(64,5)]
			connection_schemas = [
						Connection3dSchema(size=(4,4,1),strides=(2,2)),
						Connection3dSchema(size=(4,4,2),strides=(2,2)),
						Connection3dSchema(size=(4,4,4),strides=(2,2)),
						Connection3dSchema(size=(4,4,8),strides=(2,2)),
						Connection3dSchema(size=(4,4,16),strides=(2,2))]

			initial_activations = [
				lambda x: x,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu]
			activations = [
				SymmetricTanh(),
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu
				]
			initial = MultiscaleUpConv3d(feature_schemas = initial_schemas, connection_schemas = connection_schemas, activations=initial_activations)
			it1 = MultiscaleConv3d(initial_schemas, second_schemas, connection_schemas, connection_schemas, activations)
			it2 = MultiscaleConv3d(second_schemas, second_schemas, connection_schemas, connection_schemas, activations)
			it3 = MultiscaleConv3d(second_schemas, second_schemas, connection_schemas, connection_schemas, activations)
			it4 = MultiscaleConv3d(second_schemas, second_schemas, connection_schemas, connection_schemas, activations)
			it5 = MultiscaleConv3d(second_schemas, second_schemas, connection_schemas, connection_schemas, activations)

			boundary_vec = constant_variable([1,1,1,nvec_labels])
			self.boundary_vec = boundary_vec

		with tf.name_scope('trace'):
			strides2 = [(2, 2) for i in xrange(5)]
			sizes2 = [(4, 4, 1), (4, 4, 2), (4, 4, 4), (4, 4, 8), (4, 4, 16)]

			schemas2 = [
						FeatureSchema(2,0),
						FeatureSchema(24,1),
						FeatureSchema(28,2),
						FeatureSchema(32,3),
						FeatureSchema(48,4),
						FeatureSchema(64,5)]
			connection_schemas2 = [
						Connection3dSchema(size=(4,4,1),strides=(2,2)),
						Connection3dSchema(size=(4,4,2),strides=(2,2)),
						Connection3dSchema(size=(4,4,4),strides=(2,2)),
						Connection3dSchema(size=(4,4,8),strides=(2,2)),
						Connection3dSchema(size=(4,4,16),strides=(2,2))]

			initial_activations2 = [
				lambda x: x,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu]
			activations2 = [
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu
				]
			initial2 = MultiscaleUpConv3d(feature_schemas = schemas2, connection_schemas = connection_schemas2, activations=initial_activations2)
			it1_2 = MultiscaleConv3d(schemas2, schemas2, connection_schemas2, connection_schemas2, activations2)
			it2_2 = MultiscaleConv3d(schemas2, schemas2, connection_schemas2, connection_schemas2, activations2)
			it3_2 = MultiscaleConv3d(schemas2, schemas2, connection_schemas2, connection_schemas2, activations2)
			it4_2 = MultiscaleConv3d(schemas2, schemas2, connection_schemas2, connection_schemas2, activations2)
			it5_2 = MultiscaleConv3d(schemas2, schemas2, connection_schemas2, connection_schemas2, initial_activations2)

		with tf.name_scope('forward'):
			def forward(inpt):
				inpt = tf_pad_shape(inpt)
				tmp = it4(it3(it4(it3(it2(it1(initial(inpt)))))))[0]
				return tmp

		with tf.name_scope('trace'):
			def trace(seed,supervoxel):
				seed = tf_pad_shape(seed)
				supervoxel = tf_pad_shape(supervoxel)
				inpt = tf.concat(3,[seed,supervoxel])
				otpt = it5_2(it4_2(it3_2(it4_2(it3_2(it1_2(initial2(inpt)))))))[0]
				return tf.squeeze(otpt[:,:,:,0]), tf.squeeze(otpt[:,:,:,1])


		start_focus=tf.pack([tf.random_uniform([],45,55,dtype=tf.int32),
				tf.random_uniform([],400,600,dtype=tf.int32),
				tf.random_uniform([],400,600,dtype=tf.int32)])
		focuses = [start_focus]
		losses = []
		exploration_rewards = []
		label = full_human_labels[focuses[0][0],focuses[0][1],focuses[0][2]]
		commits = []
		selection_logits = []

		clear_trace = tf.assign(full_trace, tf.zeros_like(full_trace))
		with tf.control_dependencies([clear_trace]):
			for i in xrange(2):
				current_focus=focuses[-1]
				current_image = get_glimpse(full_image, current_focus)
				current_human_labels = get_glimpse(full_human_labels, current_focus)
				with tf.control_dependencies(commits):
					current_trace = get_glimpse(full_trace, current_focus)

				target = tf.to_float(tf.equal(current_human_labels, label))

				vector_labels = forward(current_image)

				supervoxel = tf.squeeze(self.affinity(vector_labels, vector_labels[16:17,78:79,78:79,:]),squeeze_dims=[3])
				new_trace_logits, proposal_logits = trace(current_trace, supervoxel)
				new_trace = tf.nn.sigmoid(new_trace_logits)
				proposal_logits=new_trace_logits
				commits.append(set_glimpse(full_trace, current_focus, new_trace))
				new_focus, selection_logit = spawn(proposal_logits, current_focus)
				focuses.append(new_focus)
				selection_logits.append(selection_logit)
				losses.append(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(new_trace_logits, target)))
				exploration_rewards.append(tf.reduce_sum(bounded_cross_entropy(new_trace, target))-tf.reduce_sum(bounded_cross_entropy(current_trace, target)))

		#our net should take as input
		#1, the seed
		#2, the spread around the centre
		#and output
		#1 the new spread
		#2 selections for where to look next

		print(selection_logits)
		print(exploration_rewards)

		loss = sum(losses) + sum([selection_logit[i]*sum(exploration_rewards[i+1:]) for i in xrange(len(selection_logits)-1)])
			
		forward_var_list = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope='params')
		trace_var_list = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope='trace')

		optimizer = tf.train.AdamOptimizer(0.002, epsilon=0.1,beta1=0.9)
		train_op = optimizer.minimize(loss, var_list = trace_var_list)

		with tf.control_dependencies([train_op]):
			train_op = tf.group(self.step.assign_add(1), tf.Print(
				0, [self.step, loss],
				message="step|loss"))
			quick_summary_op = tf.merge_summary([
				tf.scalar_summary("loss", loss),
			])

		self.summaries.append(tf.scalar_summary("loss",loss))
		summary_op = tf.merge_summary(self.summaries)

		init = tf.initialize_all_variables()
		self.sess.run(init)

		self.saver = tf.train.Saver()
		self.saver1 = tf.train.Saver(var_list=forward_var_list)
		self.train_op = train_op
		self.quick_summary_op = quick_summary_op

		self.summary_op = summary_op

	def init_log(self):
		print ('How do you want to call this experiment?')
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
		save_experiment(exp_name)
		self.summary_writer = tf.train.SummaryWriter(
			logdir, graph=self.sess.graph)

	def restore(self, modelpath, saver1=True, saver2=True):
		modelpath = os.path.expanduser(modelpath)
		if saver1:
			self.saver1.restore(self.sess, modelpath)
		if saver2:
			self.saver2.restore(self.sess, modelpath)

	def affinity(self, x, y):
		"""
		We want an array that is 1 where x and y values are the same
		and close to 0 where they are different.

		Args:
			x (tensor): 4 dimensional array of floats from 0 to 1
			y (tensor): 4 dimensional array of floats from 0 to 1

		Returns:
			tensor of floats: with ones where x == y and 0 where x ortogonal
			to y, and an smooth transition between both extreme conditions.
		"""
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

	def train(self, nsteps=100000, checkpoint_interval=250):
		self.init_log()
		print ('log initialized')
		for i in xrange(nsteps):
			t = time.time()
			if i==8:
				_, quick_summary = self.sess.run(
					[self.train_op, self.quick_summary_op], options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=self.run_metadata)
			else:
				_, quick_summary = self.sess.run(
					[self.train_op, self.quick_summary_op])

			elapsed = time.time() - t
			print("elapsed: ", elapsed)
			self.summary_writer.add_summary(
				quick_summary, self.sess.run(self.step))
			if i % checkpoint_interval == 0:
				print("checkpointing")
				step = self.sess.run(self.step)
				"""
				self.saver.save(
					self.sess,
					self.logdir +
					"model" +
					str(step) +
					".ckpt")
				"""
				self.summary_writer.add_summary(
					self.sess.run(self.summary_op), step)
				self.summary_writer.flush()
				print("done")

TRAIN = Dataset(SNEMI3D_TRAIN_DIR, image=True, human_labels=True)
args = {
	"devices": ["/gpu:0"],
	"patch_size": (32, 158, 158),
	"nvec_labels": 8,
	"maxn": 40,
	"name": "test",
	"full_image": TRAIN.image_full,
	"full_human_labels": TRAIN.human_labels_full
}
# offsets:
# devices: device used by tensorflow to run the model
# patch_size: size of window input and output by the model
# mask_boundary:
# mask_unkown
# nvec_labels: size of the vectors of every pixel
# nrecounstrctions

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)
main_model = Model(**args)

if __name__ == '__main__':
	#main_model.restore("/usr/people/jzung/experiments/adversarial/293-00-38-29-independent_discriminator/saved_models/model139287.ckpt", saver2=False)
	#main_model.restore("/usr/people/jzung/experiments/adversarial/297-22-14-02-full-loss/saved_models/model212860.ckpt",saver2=False)
	main_model.restore(pythonzenity.FileSelection(),saver2=False)
	main_model.train(nsteps=100000)
	trace = timeline.Timeline(step_stats=main_model.run_metadata.step_stats)
	trace_file = open('timeline.ctf.json', 'w')
	trace_file.write(trace.generate_chrome_trace_format())
	print("done")
