import h5py
import regiongraphs
from regiongraphs import *
import reconstruct_utils
import scipy.ndimage.measurements as measurements
import scipy.ndimage as ndimage
import numpy as np

import os
import os.path

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

class RegionCutout():
	def __init__(self, region):
		self.region=region
		self.labels = raw_labels[region]
		self.image = image[region]
		self.errors = errors[region]
		self.unique_list = filter(lambda x: x!=0, np.unique(self.labels))

	def local_errors(self, threshold=0.5):
		max_error_list = measurements.maximum(self.errors,self.labels, self.unique_list)
		additional_segments = [self.unique_list[i] for i in xrange(len(self.unique_list)) if max_error_list[i]>threshold or max_error_list[i]==0.0]
		additional_segments = filter(lambda x: x != 0, additional_segments)

		return additional_segments

def get_region(pos):
	assert all([patch_size[i]/2 < pos[i] < (full_size[i] - patch_size[i]/2) for i in range(3)])
	return tuple([slice(pos[i]-patch_size[i]/2,pos[i]+patch_size[i]-patch_size[i]/2) for i in range(3)])

def commit(traced, cutout, low_threshold=0.2, high_threshold=0.8):
	traced_list = measurements.mean(traced, cutout.labels, cutout.unique_list)
	
	assert all([x < low_threshold or x > high_threshold for x in traced_list])

	#print(zip(traced_list, unique_list))
	positive = [cutout.unique_list[i] for i in xrange(len(cutout.unique_list)) if traced_list[i]>high_threshold]
	negative = [cutout.unique_list[i] for i in xrange(len(cutout.unique_list)) if traced_list[i]<low_threshold]


	#positive = filter(lambda x: x != 0, positive)
	#negative = filter(lambda x: x != 0, negative)
	#print(positive)
	#print(negative)
	regiongraphs.add_clique(G,positive)
	regiongraphs.delete_bipartite(G,positive,negative)

def perturb(sample,radius=(1,15,15)):
	region = tuple([slice(x-y,x+y+1,None) for x,y in zip(sample,radius)])
	#print(region)
	mask = (raw_labels[region]==raw_labels[tuple(sample)]).astype(np.int32)

	patch = np.minimum(affinities[(0,)+region], mask)
	tmp=np.unravel_index(patch.argmax(),patch.shape)
	return [t+x-y for t,x,y in zip(tmp,sample,radius)]


#basename = sys.argv[1]
basename=os.path.expanduser("~/mydatasets/3_3_1/")
print("loading files...")
with h5py.File(os.path.join(basename,"samples.h5"),'r') as f:
	samples = f['main'][:]

with h5py.File(os.path.join(basename,"vertices.h5"),'r') as f:
	vertices = f['main'][:]

with h5py.File(os.path.join(basename,"edges.h5"),'r') as f:
	edges= f['main'][:]

image = h5read(os.path.join(basename,"image.h5"))
errors = h5read(os.path.join(basename,"errors3.h5"))[:]
raw_labels = h5read(os.path.join(basename,"raw.h5"))
affinities = h5read(os.path.join(basename,"aff.h5"))
#human_labels = h5read(os.path.join(basename,"proofread.h5"))
#machine_labels= h5read(os.path.join(basename,"mean_agg_tr.h5"))

print("done")

patch_size=[33,318,318]
full_size=[256,2048,2048]

G=regiongraphs.make_graph(vertices,edges)

print("sorting samples...")
nsamples = samples.shape[0]
weights = errors[[samples[:,0],samples[:,1],samples[:,2]]]

perm = np.argsort(weights)[::-1]
samples=samples[perm,:]
weights=weights[perm]
print("done")
import time

last_tic=0
def tic():
	global last_tic
	last_tic=time.time()

def toc():
	print(time.time()-last_tic)

for i in xrange(5000):
	print(i)
	try:
		tic()
		pos=perturb(samples[i,:])
		#pos = samples[i,:]
		region = get_region(pos)
		cutout=RegionCutout(region)
		toc()

		#check if segment leaves window. If not, don't grow it.
		central_segment = expand_list(G,[raw_labels[tuple(pos)]])
		central_segment_mask = reconstruct_utils.indicator(cutout.labels,central_segment)
		central_segment_bbox = ndimage.find_objects(central_segment_mask, max_label=1)[0]
		if all([x.stop-x.start < y/4 for x,y in zip(central_segment_bbox,patch_size)]):
			print("dust; not growing")
			print(central_segment_bbox)
			continue

		
		tic()
		current_segments = expand_list(G,[raw_labels[tuple(pos)]]+cutout.local_errors(threshold=0.5))
		toc()

		tic()
		mask_cutout=reconstruct_utils.indicator(cutout.labels,current_segments)
		toc()

		tic()
		t = (reconstruct_utils.flood_fill(cutout.image, mask_cutout), cutout)
		toc()

		tic()
		commit(*t)
		toc()
	except AssertionError as e:
		print(e)
h5write(os.path.join(basename,"vertices_revised.h5"),np.array(G.nodes()))
h5write(os.path.join(basename,"edges_revised.h5"),np.array(G.edges()))
