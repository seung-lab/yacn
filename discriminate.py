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
import math
import itertools
import pprint
from convkernels import *
from activations import *
import os
from datetime import datetime
from experiments import save_experiment, repo_root

import tensorflow as tf
from tensorflow.python.client import timeline

from utils import *
import dataset
from dataset import (gen, Dataset2, SNEMI3D_TRAIN_DIR, SNEMI3D_TEST_DIR, AC3_DIR, PIRIFORM_TRAIN_DIR, PIRIFORM_TEST_DIR, alternating_iterator)
import pythonzenity
from viewer import validate

class DiscrimModel(Model):
	def __init__(self, patch_size, 
				 train_data,
				 test_data,
				 train_samples,
				 test_samples,
				 devices, name=None):

		self.name=name
		self.summaries = []
		self.devices = devices
		self.patch_size = patch_size
		self.train_full_size = train_data.shape
		self.test_full_size = test_data.shape
		N=10000
		k=1000

		patchx,patchy,patchz = patch_size

		config = tf.ConfigProto(
			allow_soft_placement=True,
			gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True),
		)
		self.sess = tf.Session(config=config)
		self.run_metadata = tf.RunMetadata()

		initializer_feed_dict={}
		def scv(x):
			return static_constant_variable(x,initializer_feed_dict)

		with tf.device("/cpu:0"):
			full_labels_train = Volume(scv(train_data), patch_size)
			full_labels_test = Volume(scv(test_data), patch_size)
			validated = scv(np.zeros_like(test_data))
			train_samples = scv(train_samples)
			test_samples = scv(test_samples)

			test_running_samples = scv(np.zeros((N,3),dtype=np.int32))
			test_running_weights = scv(np.zeros((N,),dtype=np.float32))
			self.test_running_samples = test_running_samples
			self.test_running_weights = test_running_weights

			train_running_samples = scv(np.zeros((N,3),dtype=np.int32))
			train_running_weights = scv(np.zeros((N,),dtype=np.float32))
			self.train_running_samples = train_running_samples
			self.train_running_weights = train_running_weights
			
		with tf.name_scope('params'):
			self.step=tf.Variable(0)

			strides = [(2, 2) for i in xrange(5)]
			sizes = [(4, 4, 1), (4, 4, 1), (4, 4, 2), (4, 4, 4), (4, 4, 8), (4, 4, 16)]

			initial_schemas = [
						FeatureSchema(1,1),
						FeatureSchema(12,2),
						FeatureSchema(32,3),
						FeatureSchema(48,4),
						FeatureSchema(56,5),
						FeatureSchema(64,6),
						]
			connection_schemas = [
						Connection3dSchema(size=(4,4,1),strides=(2,2)),
						Connection3dSchema(size=(4,4,2),strides=(2,2)),
						Connection3dSchema(size=(4,4,4),strides=(2,2)),
						Connection3dSchema(size=(4,4,8),strides=(2,2)),
						Connection3dSchema(size=(4,4,16),strides=(2,2)),
						]

			initial_activations = [
				lambda x: x,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				]
			activations = [
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				tf.nn.elu,
				]
			#downsample = connection(FeatureSchema(1,0), FeatureSchema(4,1),
			#			Connection3dSchema(size=(4,4,1),strides=(2,2)))
			#upsample = connection(FeatureSchema(4,1), FeatureSchema(1,0),
			#		Connection3dSchema(size=(4,4,1), strides=(2,2)))
			b0=constant_variable([1],val=0.0)

			initial = MultiscaleUpConv3d(feature_schemas = initial_schemas, connection_schemas = connection_schemas, activations=initial_activations)
			it1 = MultiscaleConv3d(initial_schemas, initial_schemas, connection_schemas, connection_schemas, activations)
			it2 = MultiscaleConv3d(initial_schemas, initial_schemas, connection_schemas, connection_schemas, activations)
			it3 = MultiscaleConv3d(initial_schemas, initial_schemas, connection_schemas, connection_schemas, activations)
			it4 = MultiscaleConv3d(initial_schemas, initial_schemas, connection_schemas, connection_schemas, activations)
			it5 = MultiscaleConv3d(initial_schemas, initial_schemas, connection_schemas, connection_schemas, activations)
			w=0.1*normal_variable([1,1,1,64])
			b=constant_variable([1],val=0.0)
			extract_top = lambda x: tf.reduce_sum(w*x[-1])+b

		def draw_old():
			return tf.nn.top_k(-test_running_weights,k=k)[1][tf.random_uniform([],minval=0, maxval=k, dtype=tf.int32)]

		def draw_new():
			with tf.control_dependencies([pointer_var.assign(tf.mod(pointer_var+1,N))]):
				with tf.control_dependencies([test_running_samples[tf.identity(pointer_var),:].assign(random_row(test_samples))]):
					return tf.identity(pointer_var)

		iteration_type=tf.logical_or(rand_bool([]),tf.less(self.step, 100000*N))
		pointer_var=tf.Variable(0)
		pointer = tf.cond(iteration_type, draw_new, draw_old)
		#test_factor = tf.cond(tf.greater(self.step,N), lambda: tf.constant(2.0), lambda: tf.constant(1.0))


		focus_train=tf.reshape(random_row(train_samples),(3,))
		focus_test=tf.reshape(test_running_samples[pointer, :],(3,))
		focus_test=tf.Print(focus_test,[focus_test, pointer_var, pointer])
		
		def extract_central(inpt):
			return tf.equal(inpt, inpt[patchx/2,patchy/2,patchz/2])
		with tf.name_scope('discriminate'):
			self.cnt=0
			def discriminate(inpt):
				inpt = tf_pad_shape(random_rotation(inpt))
				self.summaries.extend([image_summary("discrim_inpt"+str(self.cnt),inpt)])
				self.cnt=self.cnt+1
				return tf.reshape(extract_top(it2(it1(initial(inpt)))),[])

			def reconstruct(inpt):
				inpt = tf_pad_shape(inpt)
				return it5(it4(it3(it2(it1(initial(inpt))))))[0]+b0

			#1 is correct and 0 is incorrect
			train_glimpse = tf.to_float(extract_central(full_labels_train[focus_train]))
			test_glimpse = tf.to_float(extract_central(full_labels_test[focus_test]))
			test_compare = tf.to_float(extract_central(full_labels_train[focus_test]))

			train_discrim = discriminate(train_glimpse)
			test_discrim = discriminate(test_glimpse)
			T,P,S = local_error(test_compare, test_glimpse)
			test_factor = tf.cond(tf.logical_and(tf.less(tf.abs(T-S),1), tf.less(tf.abs(P-S),1)), lambda:tf.constant(0.0), lambda:tf.constant(1.0))

			reconstruction = reconstruct(test_glimpse)
			#self.summaries.extend([image_summary("occluded",occluded),
			#	image_summary("reconstruction", tf.nn.sigmoid(reconstruction)),
			#	])

			#loss = tf.exp(tf.nn.relu(-train_discrim)) + tf.nn.elu(test_discrim)
			#loss = 0.5*tf.exp(tf.nn.relu(-2*train_discrim)) + test_factor * tf.nn.relu(test_discrim+10)


			loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(test_discrim, test_factor))
			reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(reconstruction, tf_pad_shape(test_compare)))


		#decision = tf.reshape(tf.to_int32(tf.py_func(validate,[extract_central(get_glimpse(full_labels_test, self.worst_focus))],[tf.bool])),[])

		#self.validate = set_glimpse(validated, self.worst_focus, tf.maximum(tf.to_int32(decision)*tf.to_int32(extract_central(get_glimpse(full_labels_test,self.worst_focus))), get_glimpse(validated, self.worst_focus)))

		var_list = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope='params')

		optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.9, epsilon=0.1)
		train_op = optimizer.minimize(1e5*loss + reconstruction_loss, var_list = var_list)

		with tf.control_dependencies([train_op]):
			with tf.control_dependencies([self.step.assign_add(1)]):
				train_op = tf.group(
						tf.Print(0, [tf.identity(self.step), loss, test_factor, T-S, P-S], message="step|loss"),
						test_running_weights[pointer].assign(test_discrim),
						train_running_weights[tf.mod(self.step,N)].assign(train_discrim),
						)
						
				quick_summary_op = tf.merge_summary([
					tf.scalar_summary("loss", loss),
					tf.scalar_summary("reconstruction_loss", reconstruction_loss),
					tf.scalar_summary("train_discrim", train_discrim),
					tf.scalar_summary("test_discrim", test_discrim),
					tf.scalar_summary("diff", train_discrim-test_discrim),
				])

		self.summaries.append(tf.scalar_summary("loss",loss))
		self.summaries.append(tf.histogram_summary("test_running_weights",test_running_weights))
		self.summaries.append(tf.histogram_summary("train_running_weights",train_running_weights))
		summary_op = tf.merge_summary(self.summaries)

		init = tf.initialize_all_variables()
		self.sess.run(init, feed_dict=initializer_feed_dict)

		self.saver = tf.train.Saver(var_list=var_list)
		self.train_op = train_op
		self.quick_summary_op = quick_summary_op
		self.summary_op = summary_op


	def get_name(self):
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
	def interrupt(self):
		return
	def export(self):
		dataset.h5write("running_samples.h5",self.sess.run(self.test_running_samples))
		dataset.h5write("running_weights.h5",self.sess.run(self.test_running_weights))

TRAIN = Dataset2(
os.path.expanduser("~/datasets/pinky_proofreading/ds/"),
{"mean_labels": "mean_labels.h5",
	"human_labels": "human_labels.h5",
	"samples": "mean_samples.h5",
	}
)
		
args = {
	"devices": ["/gpu:0"],
	"patch_size": (32, 158, 158),
	"name": "test",
	"train_data": TRAIN.human_labels.astype(np.int32),
	"train_samples": TRAIN.samples.astype(np.int32)[:,::-1]-1,
	"test_data": TRAIN.mean_labels.astype(np.int32),
	"test_samples": TRAIN.samples.astype(np.int32)[:,::-1]-1,
}

#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(args)
main_model = DiscrimModel(**args)

if __name__ == '__main__':
	#main_model.restore(pythonzenity.FileSelection())
	main_model.sess.run(main_model.step.assign(0))
	main_model.train(nsteps=1000000)
	main_model.export()
	trace = timeline.Timeline(step_stats=main_model.run_metadata.step_stats)
	trace_file = open('timeline.ctf.json', 'w')
	trace_file.write(trace.generate_chrome_trace_format())
	print("done")
