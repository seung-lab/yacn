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


def compress_labels(A,maxn):
	"""
	It relabels a segmentation so that it components go from 0 to maxn
	If there are more components that maxn it assigns the label maxn to all of them
	which is interpret as unknown.

	For each segment id in the array we replace it for the value store in d, if the sement id
	doesn't exist in d we call f to produce the next number, unless we run out of them.

	Args:
			A (numpy.array): array to relabel
			maxn (int): max number of labels to have
	
	Returns:
			numpy.array: relabeled array
	"""
	vars = [0]

	def f():
		if vars[0] <= maxn-1:
			vars[0] += 1
		return vars[0]
	d = defaultdict(f)
	d[0] = 0
	d[-1] = maxn
	ret = np.vectorize(lambda x: d[x])(A)
	if vars[0] >= maxn-3:
		print "warning, n_objects is ", vars[0]
	return ret


def pad_shape(A):
	return np.reshape(A, list(np.shape(A)) + [1])

def random_crop(patch_size, *args):
	"""
	Get a random chunk from the dataset
	The pixels close to the boundaries are a bit under represented.
	Which is probably a good idea, because the ground truth close to the boundaries
	is usually less accurate, because the tracers don't have enouch context to take
	decisions.
	
	TODO: it probably makes no difference, but why z,x,y?
	TODO: Does it make sense to replace this with 
	tf.random_crop(value, size, seed=None, name=None) ? We would probably
	want to do this operation in the cpu.
	
	Args:
			patch_size (tuple): size of the chunk to be generated, it should have 3 dimensions
			*args (arrays): one or many arrays to take chunks from
	"""
	zmax, xmax, ymax = np.shape(args[0])[0:3]
	print [np.shape(x) for x in args]
	print patch_size
	while True:
		z = np.random.randint(0, zmax - patch_size[0] + 1)
		x = np.random.randint(0, xmax - patch_size[1] + 1)
		y = np.random.randint(0, ymax - patch_size[2] + 1)
		yield [A[z:z + patch_size[0],
						 x:x + patch_size[1], 
						 y:y + patch_size[2]]
								 for A in args]

if 'LOCATION' in os.environ:
	if os.environ['LOCATION']=="tiger":
		SNEMI3D_TRAIN_DIR = os.path.expanduser("/tigress/it2/agg/SNEMI3D/train/")
		SNEMI3D_TEST_DIR = os.path.expanduser("/tigress/it2/agg/SNEMI3D/test/")
		PIRIFORM_DIR = os.path.expanduser("/tigress/it2/agg/piriform_157x2128x2128/")
		PIRIFORM_TRAIN_DIR = os.path.expanduser("/tigress/it2/agg/piriform_157x2128x2128/train/")
		PIRIFORM_TEST_DIR = os.path.expanduser("/tigress/it2/agg/piriform_157x2128x2128/test/")
	else:
		SNEMI3D_TRAIN_DIR = os.path.expanduser("~/seungmount/research/datasets/SNEMI3D/train/")
		SNEMI3D_TEST_DIR = os.path.expanduser("~/seungmount/research/datasets/SNEMI3D/test/")
		AC3_DIR = os.path.expanduser("~/seungmount/research/datasets/SNEMI3D/AC3/")
		S1_BLOCK_DIR = os.path.expanduser("~/seungmount/research/datasets/s1_block/")
		AC3_TRAIN_DIR = os.path.expanduser("~/seungmount/research/datasets/AC3/train/")
		AC3_TEST_DIR = os.path.expanduser("~/seungmount/research/datasets/AC3/test/")
		PIRIFORM_DIR = os.path.expanduser("~/seungmount/research/datasets/piriform_157x2128x2128/")
		PIRIFORM_TRAIN_DIR = os.path.expanduser("~/seungmount/research/datasets/blended_piriform_157x2128x2128/train/")
		PIRIFORM_TEST_DIR = os.path.expanduser("~/seungmount/research/datasets/blended_piriform_157x2128x2128/test/")


def load_allen_train():
	basename = os.path.expanduser("~/seungmount/Omni/TracerTasks/AIBS_practice_234251S6R_01_01_aligned_01/ground_truth/")
	train = [Dataset(basename+"vol{0:02d}/".format(i), image=True, human_labels=True, image_name = "chann.h5", human_labels_name = "seg.h5") for i in xrange(1,10)]
	test = Dataset(basename+"vol10/".format(i), image=True, human_labels=True, image_name = "chann.h5", human_labels_name = "seg.h5")
	return (train,test)

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

class Dataset():
	def __init__(self, directory, image=False, human_labels=False, machine_labels=False, semantic=False, vector=False, synapse=False, oracle_samples=False, mean_samples=False, machine_labels_name="machine_labels.h5", image_name = "image.h5", human_labels_name = "human_labels.h5", mean_samples_name = "mean_samples.h5"):
		if image:
			print directory+image_name
			self.image_full = h5read(directory + image_name).astype(np.float32)
			if self.image_full.max() > 10:
				print "dividing by 256"
				self.image_full = self.image_full/256

		if human_labels:
			print directory+human_labels_name
			self.human_labels_full = h5read(directory + human_labels_name).astype(np.int32)

		if machine_labels:
			self.machine_labels_full = h5read(directory + machine_labels_name).astype(np.int32)

		if semantic:
			self.semantic_full = h5read(directory + "human_semantic_labels.h5").astype(np.int32)

		if mean_samples:
			self.mean_samples_full = h5read(directory + mean_samples_name)

		if oracle_samples:
			self.oracle_samples_full = h5read(directory + "oracle_samples.h5")
		
		def select_synapse(x):
			if x==3 or x==6:
				return 1e0
			else:
				return 0e0
		
		if synapse:
			synapse_full = h5read(directory + "human_synapse_labels.h5").astype(np.uint32)
			synapse_full = np.vectorize(select_synapse)(synapse_full)
		
		if vector:
			self.vector_full = np.transpose(h5read(directory + "vector_labels.h5").astype(np.float32),axes=[1,2,3,0])

def gen(dataset, patch_size, maxn):
	"""
	It randomly permutes x and y and indepently of that it randomly flips z,y and x.
	Everything with probability .5 .

	TODO: Maybe add more data augmentation, by modifying individual pixel values,
	or elastically modifying the image, or whatever lastest kisuk's tricks are.
	
	TODO: rename this to augment_chunks
	Returns:
			list: where the first item is an array of EM images, and the second one are labels
	"""
	if type(dataset)==type([]):
		for x in alternating_iterator([gen(x,patch_size,maxn) for x in dataset],[1 for x in dataset]):
			yield x

	for image, human_labels in random_crop(patch_size, dataset.image_full, dataset.human_labels_full):
		r1 = random.choice([slice(0,None,1),slice(-1,None,-1)])
		r2 = random.choice([slice(0,None,1),slice(-1,None,-1)])
		r3 = random.choice([slice(0,None,1),slice(-1,None,-1)])
		perm = random.choice([[0,1,2],[0,2,1]])

		transform = lambda x: np.transpose(x[r1,r2,r3],axes=perm)
		yield map(transform,(image, compress_labels(human_labels,maxn)))

def alternating_iterator(its, counts, label=False):
	while True:
		for k,(it, count) in enumerate(zip(its, counts)):
			for i in xrange(count):
				if label:
					yield (k,it.next())
				else:
					yield it.next()


def gen_completion_pairs(directory):
	l=os.listdir(directory)
	def transform(x):
		return np.transpose(x[r1,r2,r3],axes=perm)
	while True:
		path=os.path.join(directory,random.choice(l))
		f = h5py.File(path, "r")
		inpt = f["input"][()].astype(np.float32)
		otpt = f["output"][()].astype(np.float32)
		f.close()

		r1 = random.choice([slice(0,None,1),slice(-1,None,-1)])
		r2 = random.choice([slice(0,None,1),slice(-1,None,-1)])
		r3 = random.choice([slice(0,None,1),slice(-1,None,-1)])
		perm = random.choice([[0,1,2],[0,2,1]])

		yield map(transform,(inpt,otpt))
