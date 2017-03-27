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

	def __init__(self, patch_size,
				 nvec_labels, maxn,
				 devices, name=None):

		self.name=name
		self.summaries = []
		self.devices = devices
		self.patch_size = patch_size
		self.padded_patch_size = (1,) + patch_size + (1,)
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

		with tf.name_scope('params'):
			self.step = tf.Variable(0)
			forward = basic_net2.make_forward_net(patch_size, 2, nvec_labels)

		params_var_list = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope='params')

		self.image_feed =tf.placeholder(tf.float32, shape=self.padded_patch_size)
		self.mask_feed = tf.placeholder(tf.float32, shape=self.padded_patch_size)
		self.vector_labels_test = forward(tf.concat([self.image_feed, self.mask_feed],4))
		self.expansion_test = affinity(extract_central(self.vector_labels_test), self.vector_labels_test)

		init = tf.global_variables_initializer()
		self.sess.run(init)

		self.saver = tf.train.Saver(var_list=params_var_list)

	def get_filename(self):
		return os.path.splitext(os.path.basename(__file__))[0]

	def test(self, image, mask):
		image = dataset.prep("image",image)
		mask = dataset.prep("image",mask)
		ret = self.sess.run(self.expansion_test, feed_dict={self.image_feed: image, self.mask_feed: mask})
		return ret

patch_size=(33,318,318)
args = {
	"devices": get_device_list(),
	"patch_size": patch_size,
	"nvec_labels": 6,
	"maxn": 40,
	"name": "test",
}

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)
main_model = VectorLabelModel(**args)
#main_model.restore(zenity_workaround())
