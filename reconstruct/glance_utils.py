import h5py
import numpy as np
from multiprocessing import Process, Queue
import os
import misc_utils
import uuid

class FileArray(object):
	def __init__(self, A):
		self.filename = "/tmp/" + str(uuid.uuid4().hex) + ".h5"
		misc_utils.h5write(self.filename, A)
	
	def get(self):
		print("reading from " + self.filename)
		return misc_utils.h5read(self.filename,force=True)
def unpack(A):
	if type(A) == FileArray:
		return A.get()
	else:
		return A
def pack(A):
	if type(A) == np.ndarray:
		return FileArray(A)
	else:
		return A

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
	sys.path.insert(0, os.path.expanduser("~/nets/nets"))
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
	sys.path.insert(0, os.path.expanduser("~/nets/nets"))
	import discriminate2_online_inference
	discriminate2_online_inference.main_model.restore("~/checkpoint/discriminate2/model887401.ckpt")
	print("ready")

	while True:
		try:
			#print("waiting")
			mask = q1.get()[0]
			#print("received")
			q2.put(pack(discriminate2_online_inference.main_model.test(mask)))
			#print("done")
		except Exception as e:
			print(e)

def run_recompute_discrim(q1,q2):
	import os
	os.environ["CUDA_VISIBLE_DEVICES"]="1"
	import sys
	sys.path.insert(0, os.path.expanduser("~/nets/nets"))
	import discriminate2_inference
	discriminate2_inference.__init__([1,256,1024,1024,1],checkpoint="~/seung1001_experiments/discriminate2/081-19-27-58-test/model362700.ckpt")
	while True:
		try:
			#print("waiting")
			seg, samples, err, visited = map(unpack,q1.get())
			X,Y,Z=np.shape(seg)
			#print("received")
			q2.put(np.reshape(discriminate2_inference.main_model.inference(seg,samples, visited=visited,ret=err), [X,Y,Z]))
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
