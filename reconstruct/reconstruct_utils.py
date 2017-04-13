import numpy as np
import sys
import os
import os.path
import h5py
import numpy as np
from multiprocessing import Process, Queue
import misc_utils
import uuid
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def run_trace(q1,q2,device):
	import os
	os.environ["CUDA_VISIBLE_DEVICES"]=device

	import sys
	sys.path.insert(0, os.path.expanduser("~/yacn/nets"))
	import sparse_vector_labels_inference
	#import pythonzenity
	sparse_vector_labels_inference.main_model.restore("~/experiments/sparse_vector_labels/latest.ckpt")
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

def run_discrim_online(q1,q2,device):
	import os
	os.environ["CUDA_VISIBLE_DEVICES"]=device

	import sys
	sys.path.insert(0, os.path.expanduser("~/yacn/nets"))
	import discriminate3_online_inference
	#import pythonzenity
	discriminate3_online_inference.main_model.restore("~/experiments/discriminate3/latest.ckpt")
	print("ready")
	while True:
		try:
			#print("waiting")
			image, mask = q1.get()
			X,Y,Z=np.shape(image)
			#print("received")
			q2.put(np.reshape(discriminate3_online_inference.main_model.test(image, mask),[X,Y,Z]))
			#print("done")
		except Exception as e:
			print(e)


def run_recompute_discrim(q1,q2,device):
	import os
	os.environ["CUDA_VISIBLE_DEVICES"]=device
	import sys
	sys.path.insert(0, os.path.expanduser("~/nets/nets"))
	import discriminate2_inference
	discriminate3_inference.__init__([1,256,2048,1048,1],checkpoint="~/experiments/discriminate3/latest.ckpt")
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
	def __init__(self,f,device):
		self.q1 = Queue()
		self.q2 = Queue()
		self.p = Process(target=f, args=(self.q1,self.q2,device))
		self.p.daemon=True
		self.p.start()
	def __call__(self, *args):
		self.q1.put(args)
		return self.q2.get()


discrim_online_daemon = ComputeDaemon(run_discrim_online,"0")
trace_daemon = ComputeDaemon(run_trace,"1")
#discrim_daemon = ComputeDaemon(run_recompute_discrim,"1")
