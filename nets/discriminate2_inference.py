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
	def __init__(self, patch_size, full_size,
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

		initializer_feed_dict={}
		self.ret_placeholder=tf.placeholder(dtype=tf.float32,shape=full_size)
		self.machine_labels_placeholder=tf.placeholder(dtype=tf.int32,shape=full_size)
		self.visited_placeholder=tf.placeholder(dtype=tf.int32,shape=full_size)
	
		self.ret = tf.Variable(self.ret_placeholder)
		self.visited = tf.Variable(self.visited_placeholder)
		self.machine_labels = tf.Variable(self.machine_labels_placeholder)

		ret_vol = Volume(self.ret, self.padded_patch_size)
		visited_vol = Volume(self.visited, self.padded_patch_size)
		machine_labels_vol = Volume(self.machine_labels, self.padded_patch_size)

		self.focus_inpt = tf.placeholder(shape=(3,),dtype=tf.int32)
		focus=tf.concat([[0],self.focus_inpt,[0]],0)

		machine_labels_glimpse = equal_to_centre(machine_labels_vol[focus])
		coverage_mask = tf.ones_like(machine_labels_glimpse)

		with tf.name_scope('iteration'):
			def f():
				with tf.device("/gpu:0"):
					discrim_tower = self.discrim(machine_labels_glimpse)
				i=3
				ds_shape = static_shape(discrim_tower[i])
				print(ds_shape)
				expander = compose(*reversed(discrim_net.range_expanders[0:i]))
				otpt = upsample_max(tf.nn.sigmoid(discrim_tower[i]), self.padded_patch_size, expander)
				
				with tf.control_dependencies([
						ret_vol.__setitem__(focus,tf.maximum(coverage_mask * otpt * machine_labels_glimpse,ret_vol[focus])), 
						visited_vol.__setitem__(focus,tf.add(tf.to_int32(coverage_mask * machine_labels_glimpse), visited_vol[focus]))
						]):
					return tf.no_op()

			self.it = tf.cond(self.visited[focus[0],focus[1],focus[2],focus[3],focus[4]] > 3,
					lambda: tf.no_op(),
					f)

		self.full_array_initializer = tf.variables_initializer([self.ret,self.visited,self.machine_labels])

		self.sess.run(self.full_array_initializer, feed_dict={
			self.machine_labels_placeholder: np.zeros(full_size, dtype=np.int32), 
			self.ret_placeholder: np.zeros(full_size,dtype=np.float32), 
			self.visited_placeholder: np.zeros(full_size,dtype=np.int32)})


		var_list = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope='params')

		self.sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='iteration')),feed_dict={self.focus_inpt:[0,0,0]})

		self.saver = tf.train.Saver(var_list=var_list)
	
	#plan: assign to each object the magnitude of the max value of the error detector in the window.
	#vol should be machine_labels of size [1,X,Y,Z,1]
	def inference(self, machine_labels, sample_generator, ret=None, visited=None):
		if type(sample_generator) == np.ndarray:
			sample_generator = random_sample_generator(sample_generator)
		machine_labels = dataset.prep("machine_labels", machine_labels)
		ret = dataset.prep("errors", ret)
		visited = dataset.prep("machine_labels", visited)

		if ret is None:
			ret = np.zeros_like(machine_labels, dtype=np.float32)
		if visited is None:
			visited = np.zeros_like(machine_labels, dtype=np.int32)
		self.sess.run(self.full_array_initializer, feed_dict={
			self.machine_labels_placeholder: machine_labels, 
			self.ret_placeholder: ret, 
			self.visited_placeholder: visited})
		
		for sample in sample_generator:
			_= self.sess.run(self.it, feed_dict={self.focus_inpt: sample})

		return self.sess.run(self.ret)

	def get_filename(self):
		return os.path.splitext(os.path.basename(__file__))[0]

#samples should be a (N,3) array
def random_sample_generator(samples,k=None):
	N=samples.shape[0]
	if k is None:
		k=N
	for i in random.sample(range(N),k):
		print(i)
		yield samples[i,:]

def __init__(full_size, checkpoint=None):
	global main_model
	args = {
		"devices": get_device_list(),
		"patch_size": tuple(discrim_net.patch_size_suggestions([2,3,3])[0]),
		"full_size": full_size,
		"name": "test",
	}
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(args)
	with tf.device("/cpu:0"):
		main_model = DiscrimModel(**args)
	if checkpoint is None:
		main_model.restore(zenity_workaround())
	else:
		main_model.restore(checkpoint)
	print("model initialized")

if __name__ == '__main__':
	TRAIN = MultiDataset(
			[
				#os.path.expanduser("~/seungmount/research/ranl/error_detector/ds/"),
				os.path.expanduser("~/mydatasets/3_3_1/ds/"),
			],
			{
				"machine_labels": "mean_agg_tr.h5",
				"samples": "samples.h5",
			}
	)
	__init__(full_size=tuple(TRAIN.machine_labels[0].shape))

	dataset.h5write(os.path.join(TRAIN.directories[0], "errors4.h5"), 
			np.squeeze(
				main_model.inference(
					TRAIN.machine_labels[0], 
					sample_generator = random_sample_generator(TRAIN.samples[0])),
				axis=(0,4)))
