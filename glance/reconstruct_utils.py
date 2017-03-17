import numpy as np
import sys
import os
import os.path

sys.path.insert(0, os.path.expanduser("~/nets"))
import sparse_vector_labels_inference
sparse_vector_labels_inference.main_model.restore("~/experiments/sparse_vector_labels/trained_057-23-23-56-test/model999800.ckpt")

def flood_fill(image, mask):
	X,Y,Z=np.shape(image)
	return np.reshape(sparse_vector_labels_inference.main_model.test(image, mask),[X,Y,Z])

def indicator(A, s):
	return np.reshape(np.in1d(A,np.array(list(s))).astype(np.int32),np.shape(A))
