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
from dataset import (gen, Dataset2, alternating_iterator)
import pythonzenity
from viewer import validate

class DiscrimModel(Model):
	def __init__(self, patch_size, 
				 train_data,
				 test_data,
				 devices, name=None):

		self.name=name
		self.summaries = []
		self.devices = devices
		self.patch_size = patch_size
		self.train_full_size = train_data.shape
		self.test_full_size = test_data.shape

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
			full_human_labels = scv(train_data)
			full_machine_labels = scv(test_data)
		self.focus= tf.placeholder(tf.int32, shape=(3,))
	
		def extract_central(inpt):
			return tf.equal(inpt, inpt[patchx/2,patchy/2,patchz/2])

		def get_glimpse(A, focus):
			with tf.device("/cpu:0"):
				corner = focus - np.array([x/2 for x in patch_size],dtype=np.int32)
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
		human_glimpse = tf.to_float(extract_central(get_glimpse(full_human_labels, self.focus)))
		machine_glimpse = tf.to_float(extract_central(get_glimpse(full_machine_labels, self.focus)))

		self.comparison = local_error(human_glimpse, machine_glimpse)

		init = tf.initialize_all_variables()
		self.sess.run(init, feed_dict=initializer_feed_dict)

TRAIN = Dataset2(
os.path.expanduser("~/datasets/pinky_proofreading/ds/"),
{"mean_labels": "mean_labels.h5",
	"human_labels": "human_labels.h5",
	"samples": "mean_samples.h5",
	}
)
args = {
	"devices": ["/gpu:0"],
	"patch_size": (48, 314, 314),
	"name": "test",
	"train_data": TRAIN.human_labels.astype(np.int32),
	"test_data": TRAIN.mean_labels.astype(np.int32),
}
main_model = DiscrimModel(**args)

if __name__ == '__main__':
	samples = TRAIN.samples.astype(np.int32)[:,::-1]-1
	N=samples.shape[0]
	otpt = np.zeros((N,3),dtype=np.float32)
	for i in xrange(0,N):
		print(str(i)+"/"+str(N))
		otpt[i,:]=main_model.sess.run(main_model.comparison,feed_dict={main_model.focus:samples[i,:]})
	dataset.h5write("sample_data.h5",otpt)
