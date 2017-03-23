import h5py
import numpy as np
from multiprocessing import Process, Queue
import os

def visualize_colour(viewer,filename):
	"""
	This is holding all the affinities in RAM,
	it would be easy to modify so that it is
	reading from disk directly.
	"""
	try:
		with h5py.File(filename,'r') as f:
			data = f['main'][:,:,:,:].astype(np.float32)
			viewer.add(data, name=filename, shader="""
			void main() {
			  emitRGB(
					vec3(toNormalized(getDataValue(0)),
					toNormalized(getDataValue(1)),
					toNormalized(getDataValue(2))
						 )
					  );
			}
			""")
	except IOError:
		print(filename+' not found')


def run_trace(q1,q2):
	import os
	os.environ["CUDA_VISIBLE_DEVICES"]="0"

	import sys
	sys.path.insert(0, os.path.expanduser("~/nets"))
	import sparse_vector_labels_inference
	#import pythonzenity
	sparse_vector_labels_inference.main_model.restore("~/experiments/sparse_vector_labels/057-23-23-56-test/model999800.ckpt")
	print("ready")
	while True:
		try:
			#print("waiting")
			image, mask = q1.get()
			X,Y,Z=np.shape(image)
			#print("received")
			q2.put(np.reshape(sparse_vector_labels_inference.main_model.test(image, mask),[X,Y,Z]))
			#print("done")
		except Exception as e:
			print(e)

def run_discrim(q1,q2):
	import os
	os.environ["CUDA_VISIBLE_DEVICES"]="1"
	import sys
	sys.path.insert(0, os.path.expanduser("~/nets"))
	import discriminate2_online_inference
	discriminate2_online_inference.main_model.restore("~/checkpoint/discriminate2/model887401.ckpt")
	print("ready")

	while True:
		try:
			#print("waiting")
			mask = q1.get()[0]
			#print("received")
			q2.put(discriminate2_online_inference.main_model.test(mask))
			#print("done")
		except Exception as e:
			print(e)

class ComputeDaemon():
	def __init__(self,f):
		self.q1 = Queue()
		self.q2 = Queue()
		self.p = Process(target=f, args=(self.q1,self.q2))
		self.p.daemon=True
		self.p.start()
	def __call__(self, *args):
		self.q1.put(args)
		return self.q2.get()

def segment_means(A,labels):
	d=defaultdict(lambda: 0)
	count=defaultdict(lambda: 0)
	for i in np.ndindex(A.shape):
		d[labels[i]]+=A[i]
		count[labels[i]]+=1
	for key in d:
		d[key] /= count[key]

	return d

def indicator(A, s):
	return np.reshape(np.in1d(A,np.array(list(s))).astype(np.int32),np.shape(A))
