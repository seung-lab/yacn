
from __future__ import print_function
import time
import h5py
import numpy as np
files = []
def h5read(filename, force=False):
	try:
		if force:
			with h5py.File(filename,'r') as f:
				return f['main'][:]
		else:
			f=h5py.File(filename,'r')
			global files
			files.append(f)
			return f['main']
	except IOError:
		print(filename+' not found')

def h5write(filename, x):
	f = h5py.File(filename, "w")
	dset = f.create_dataset("main", data=x)
	f.close()

tics=[]
def tic():
	global tics
	tics.append(time.time())

def toc(msg="toc"):
	elapsed = time.time() - tics.pop()
	print("\t"*len(tics) + msg + " " + str(elapsed))

def indicator(A, s):
	return np.reshape(np.in1d(A,np.array(list(s))).astype(np.int32),np.shape(A))
