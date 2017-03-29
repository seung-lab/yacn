import numpy as np
import sys
import os
import os.path
import glance_utils

"""
sys.path.insert(0, os.path.expanduser("~/nets"))
import sparse_vector_labels_inference
sparse_vector_labels_inference.main_model.restore("~/experiments/sparse_vector_labels/trained_057-23-23-56-test/model999800.ckpt")
"""

discrim_daemon = glance_utils.ComputeDaemon(glance_utils.run_recompute_discrim)
trace_daemon = glance_utils.ComputeDaemon(glance_utils.run_trace)

def flood_fill(image, mask):
	"""
	X,Y,Z=np.shape(image)
	return np.reshape(sparse_vector_labels_inference.main_model.test(image, mask),[X,Y,Z])
	"""
	return trace_daemon(image,mask)

def discrim(mask):
	subsampled_mask = mask[:,1:-1:2,1:-1:2]
	X,Y,Z=np.shape(subsampled_mask)
	ret=np.zeros_like(mask,dtype=np.float32)
	ret[:,1:-1:2,1:-1:2]=np.reshape(discrim_daemon(subsampled_mask),[X,Y,Z])
	return ret

def indicator(A, s):
	return np.reshape(np.in1d(A,np.array(list(s))).astype(np.int32),np.shape(A))