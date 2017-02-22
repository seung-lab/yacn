import h5py
import numpy as np
import random
import itertools
#import dataprep
from collections import defaultdict
from multiprocessing import Process, Queue
import os
import random


def h5read(filename):
	f = h5py.File(filename, "r")
	tmp = f["main"][()]
	f.close()
	return tmp


def h5write(filename, x):
	f = h5py.File(filename, "w")
	dset = f.create_dataset("main", data=x)
	f.close()

class Dataset2():
	def __init__(self, directory,d):
		self.directory=directory
		for (k,v) in d.items():
			setattr(self, k, h5read(directory + v))
		if hasattr(self, "image"):
			self.image = self.image.astype(np.float32)
			if self.image.max() > 10:
				print "dividing by 256"
				self.image = self.image/256

		if hasattr(self, "human_labels"):
			self.human_labels = self.human_labels.astype(np.int32)

		if hasattr(self, "machine_labels"):
			self.machine_labels = self.machine_labels.astype(np.int32)

def alternating_iterator(its, counts, label=False):
	while True:
		for k,(it, count) in enumerate(zip(its, counts)):
			for i in xrange(count):
				if label:
					yield (k,it.next())
				else:
					yield it.next()
