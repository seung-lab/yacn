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
			n_volumes = len(truth_data)
			full_labels_truth = MultiVolume(map(scv,truth_data), self.padded_patch_size)
			full_labels_lies = MultiVolume(map(scv,lies_data), self.padded_patch_size)
			samples = MultiVolume(map(scv,samples), (1,3), indexing='CORNER')
	
		with tf.name_scope('params'):
			self.step=tf.Variable(0)
			discrim, reconstruct = discrim_net.make_forward_net(patch_size,1,1)
			self.discrim = discrim

		#for some reason we need to initialize here first... Figure this out!
		init = tf.global_variables_initializer()
		self.sess.run(init, feed_dict=initializer_feed_dict)

		print("initialized1")
		loss=0
		reconstruction_loss=0
		for i,d in enumerate(devices):
			with tf.device(d):
				with tf.name_scope("gpu"+str(i)):
					vol_id=tf.random_uniform([], minval=0, maxval=n_volumes, dtype=np.int32)
					focus_lies=tf.concat([[0],tf.reshape(samples[vol_id,('RAND',0)],(3,)),[0]],0)
					focus_lies = tf.Print(focus_lies, [vol_id, focus_lies], summarize=10)
			
					rr=RandomRotationPadded()

					#1 is correct and 0 is incorrect
					#truth_glimpse = rr(equal_to_centre(full_labels_truth[vol_id,focus_truth]))
					lies_glimpse = rr(equal_to_centre(full_labels_lies[vol_id,focus_lies]))
					tmp = full_labels_truth[vol_id,focus_lies]
					truth_glimpse = rr(equal_to_centre(tmp))
					human_labels = rr(tmp)
					
					self.summaries.append(image_summary("lies_glimpse", lies_glimpse))
					self.summaries.append(image_summary("truth_glimpse", truth_glimpse))
					self.summaries.append(image_summary("human_labels", tf.to_float(human_labels)))
					
					occluded = random_occlusion(lies_glimpse)
					reconstruction = reconstruct(occluded)
					reconstruction_loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=truth_glimpse))
					
					self.summaries.append(image_summary("reconstruction", tf.nn.sigmoid(reconstruction)))
					self.summaries.append(image_summary("occluded", occluded))

					truth_discrim_tower = discrim(truth_glimpse)
					lies_discrim_tower = discrim(lies_glimpse)

					for i in range(3,5):
						with tf.device("/cpu:0"):
							ds_shape = static_shape(lies_discrim_tower[i])
							expander = compose(*reversed(discrim_net.range_expanders[0:i]))

							print(ds_shape)
							tmp=slices_to_shape(expander(shape_to_slices(ds_shape[1:4])))
							assert tuple(tmp) == tuple(self.patch_size)
							errors = localized_errors(lies_glimpse, human_labels, ds_shape = ds_shape, expander=expander)
							errors = tf.Print(errors, [tf.reduce_sum(errors)])
							loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = lies_discrim_tower[i], labels=errors))
							loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = truth_discrim_tower[i], labels=tf.zeros_like(truth_discrim_tower[i])))
							self.summaries.append(image_summary("guess"+str(i), upsample_mean(tf.nn.sigmoid(lies_discrim_tower[i]), self.padded_patch_size, expander)))
							self.summaries.append(image_summary("truth"+str(i), upsample_mean(errors, self.padded_patch_size, expander)))

					loss += tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reduce_sum(lies_discrim_tower[-1]), labels=has_error(lies_glimpse, human_labels))
					loss += tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reduce_sum(truth_discrim_tower[-1]), labels=tf.constant(0,dtype=tf.float32))



		var_list = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope='params')

		optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, epsilon=0.1)
		train_op = optimizer.minimize(1e5*loss + reconstruction_loss, colocate_gradients_with_ops=True, var_list = var_list)

		with tf.control_dependencies([train_op]):
			with tf.control_dependencies([self.step.assign_add(1)]):
				train_op = tf.group(
						tf.Print(0, [tf.identity(self.step), loss], message="step|loss"),
						)
						
				quick_summary_op = tf.summary.merge([
					tf.summary.scalar("loss", loss),
					tf.summary.scalar("reconstruction_loss", reconstruction_loss),
				])

		self.summaries.append(tf.summary.scalar("loss",loss))
		summary_op = tf.summary.merge(self.summaries)

		init = tf.global_variables_initializer()
		self.sess.run(init, feed_dict=initializer_feed_dict)

		self.saver = tf.train.Saver(var_list=var_list)
		self.train_op = train_op
		self.quick_summary_op = quick_summary_op
		self.summary_op = summary_op
	
	def get_filename(self):
		return os.path.splitext(os.path.basename(__file__))[0]

	def interrupt(self):
		return
	def train_feed_dict(self):
		return {}

TRAIN = MultiDataset(
		[
			os.path.expanduser("~/mydatasets/1_1_1/ds/"),
			#os.path.expanduser("~/mydatasets/1_2_1/ds/"),
			#os.path.expanduser("~/mydatasets/2_1_1/ds/"),
			#os.path.expanduser("~/mydatasets/2_2_1/ds/"),
		],
		{
			"machine_labels": "mean_agg_tr.h5",
			"human_labels": "proofread.h5",
			"samples": "samples.h5",
		}
)
		
args = {
	"devices": get_device_list(),
	"patch_size": tuple(discrim_net.patch_size_suggestions([2,3,3])[0]),
	"name": "test",
	"truth_data": TRAIN.human_labels,
	"lies_data": TRAIN.machine_labels,
	"samples": TRAIN.samples,
}

#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(args)
#with tf.device(args["devices"][0]):
main_model = DiscrimModel(**args)
main_model.restore(zenity_workaround())
print("model initialized")
if __name__ == '__main__':
	main_model.train(nsteps=1000000, checkpoint_interval=200)
