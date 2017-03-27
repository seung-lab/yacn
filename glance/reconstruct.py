from __future__ import print_function
import regiongraphs
from regiongraphs import *
import reconstruct_utils
import scipy.ndimage.measurements as measurements
import scipy.ndimage as ndimage
import numpy as np
import pandas as pd
import misc_utils
import sys
import traceback
from misc_utils import *
import glance_utils

import os
import os.path

class ReconstructionException(Exception):
	pass

class RegionCutout():
	def __init__(self, region):
		self.region=region
	def get_raw_labels(self):
		if not hasattr(self, 'raw_labels'):
			self.raw_labels = raw_labels[self.region]
		return self.raw_labels
	def get_changed(self):
		if not hasattr(self, 'changed'):
			self.changed = changed[self.region]
		return self.changed
	def get_image(self):
		if not hasattr(self, 'image'):
			self.image= image[self.region]
		return self.image
	def get_human_labels(self):
		if not hasattr(self, 'human_labels'):
			self.human_labels = human_labels[self.region]
		return self.human_labels
	def get_labels(self):
		if not hasattr(self, 'machine_labels'):
			self.machine_labels = machine_labels[self.region]
		return self.machine_labels
	def get_errors(self):
		if not hasattr(self, 'errors'):
			self.errors= errors[region]
		return self.errors
	def get_unique_list(self):
		if not hasattr(self, 'unique_list'):
			self.unique_list = filter(lambda x: x!=0, np.unique(self.get_raw_labels()))
		return self.unique_list
	def get_subgraph(self):
		if not hasattr(self, 'subgraph'):
			global G
			self.subgraph = G.subgraph(self.get_unique_list())
		return self.subgraph
	def get_local_labels(self):
		if not hasattr(self, 'local_labels'):
			tic()
			components=nx.connected_components(self.get_subgraph())
			d={}
			for i,nodes in enumerate(components,1):
				for node in nodes:
					d[node]=i
			d[0]=0
			self.local_labels = np.vectorize(d.get)(self.raw_labels)
			toc("local labels")
		return self.local_labels

	def local_errors(self, threshold=0.5):
		unique_list = self.get_unique_list()
		max_error_list = measurements.maximum(self.get_errors(),self.get_raw_labels(), unique_list)
		additional_segments = [unique_list[i] for i in xrange(len(unique_list)) if max_error_list[i]>threshold or max_error_list[i]==0.0]
		additional_segments = filter(lambda x: x != 0, additional_segments)

		return additional_segments

def get_region(pos):
	if not all([patch_size[i]/2 < pos[i] < (full_size[i] - patch_size[i]/2) for i in range(3)]):
		raise ReconstructionException("out of bounds")
	return tuple([slice(pos[i]-patch_size[i]/2,pos[i]+patch_size[i]-patch_size[i]/2) for i in range(3)])

datalist=pd.DataFrame([],columns=["guess","truth","volume","seg_id","example_id"])

def analyze(traced, cutout,example_id):
	args = [cutout.get_local_labels(), filter(lambda x: x!=0, np.unique(cutout.get_local_labels()))]
	guess = measurements.mean(traced, *args)
	truth = measurements.mean(cutout.get_human_labels()[np.unravel_index(np.argmax(traced),cutout.get_raw_labels().shape)]==cutout.get_human_labels(), *args)
	volumes = measurements.sum(np.ones_like(cutout.get_raw_labels()), *args)

	global datalist
	datalist=datalist.append(pd.DataFrame([[guess[i],truth[i],volumes[i],args[1][i],example_id] for i in xrange(len(args[1]))],columns=["guess","truth","volume","seg_id","example_id"]))

def commit(traced, cutout, low_threshold=0.2, high_threshold=0.8):
	unique_list = cutout.get_unique_list()
	traced_list = measurements.mean(traced, cutout.get_raw_labels(), cutout.get_unique_list())
	
	if not all([x < low_threshold or x > high_threshold for x in traced_list]):
		raise ReconstructionException("not confident")

	#print(zip(traced_list, unique_list))
	positive = [unique_list[i] for i in xrange(len(unique_list)) if traced_list[i]>high_threshold]
	negative = [unique_list[i] for i in xrange(len(unique_list)) if traced_list[i]<low_threshold]

	#print(positive)
	#print(negative)
	original_components = list(nx.connected_components(G.subgraph(unique_list)))
	regiongraphs.add_clique(G,positive)
	regiongraphs.delete_bipartite(G,positive,negative)
	new_components = list(nx.connected_components(G.subgraph(unique_list)))
	changed_list = set(unique_list) - set.union(*([set([])]+[s for s in original_components if s in new_components]))
	changed_cutout = indicator(cutout.get_raw_labels(),  changed_list)
	changed[cutout.region] = np.maximum(changed[cutout.region], changed_cutout)

def perturb(sample,radius=(1,15,15)):
	region = tuple([slice(x-y,x+y+1,None) for x,y in zip(sample,radius)])
	#print(region)
	mask = (raw_labels[region]==raw_labels[tuple(sample)]).astype(np.int32)

	patch = np.minimum(affinities[(0,)+region], mask)
	tmp=np.unravel_index(patch.argmax(),patch.shape)
	return [t+x-y for t,x,y in zip(tmp,sample,radius)]

def flatten(G, raw):
	components = nx.connected_components(G)
	d={}
	for i,nodes in enumerate(components,1):
		for node in nodes:
			d[node]=i
	d[0]=0
	
	mp = np.arange(0,max(d.keys())+1)
	mp[d.keys()] = d.values()
	return mp[raw]
	#return np.vectorize(d.get)(raw)

def recompute_errors(epoch=None):
	global changed
	global errors
	global raw_labels
	global samples
	print("preparing to recompute errors")
	sub_errors = np.minimum(errors[:,::2,::2], 1-changed[:,::2,::2])
	sub_visited = 4*(1 - changed[:, ::2, ::2])
	sub_raw_labels = raw_labels[:,::2,::2]
	sub_samples = np.array(filter(lambda i: sub_visited[i[0],i[1],i[2]]==0, ds_samples))
	print(sub_samples.shape)
	print("flattening current seg")
	sub_machine_labels = flatten(G, sub_raw_labels)

	print("recomputing errors")
	sub_new_errors = glance_utils.unpack(reconstruct_utils.discrim_daemon(*(map(glance_utils.pack,[sub_machine_labels, sub_samples, sub_errors, sub_visited]))))
	
	errors = np.zeros_like(errors)
	errors[:,::2,::2] = sub_new_errors

	print("logging")
	if epoch is None:
		name = "epoch"
	else:
		name = "epoch" + str(epoch)
	h5write(os.path.join(basename,name+"_errors.h5"),sub_new_errors)
	h5write(os.path.join(basename,name+"_changed.h5"),changed[:,::2,::2])
	h5write(os.path.join(basename,name+"_machine_labels.h5"),sub_machine_labels)

def sort_samples():
	global nsamples
	global weights
	nsamples = samples.shape[0]
	weights = errors[[samples[:,0],samples[:,1],samples[:,2]]]


#basename = sys.argv[1]
basename=os.path.expanduser("~/mydatasets/3_3_1/")
print("loading files...")
with h5py.File(os.path.join(basename,"samples.h5"),'r') as f:
	samples = f['main'][:]

with h5py.File(os.path.join(basename,"ds/samples.h5"),'r') as f:
	ds_samples = f['main'][:]

with h5py.File(os.path.join(basename,"vertices.h5"),'r') as f:
	vertices = f['main'][:]

with h5py.File(os.path.join(basename,"edges.h5"),'r') as f:
	edges= f['main'][:]

image = h5read(os.path.join(basename,"image.h5"))
errors = h5read(os.path.join(basename,"errors3.h5"))[:]
raw_labels = h5read(os.path.join(basename,"raw.h5"))
affinities = h5read(os.path.join(basename,"aff.h5"))
human_labels = h5read(os.path.join(basename,"proofread.h5"))
machine_labels= h5read(os.path.join(basename,"mean_agg_tr.h5"))
changed = np.zeros_like(machine_labels, dtype=np.int32)

print("done")

patch_size=[33,318,318]
full_size=[256,2048,2048]

G=regiongraphs.make_graph(vertices,edges)

print("sorting samples...")
sort_samples()

perm = np.argsort(weights)[::-1]
samples=samples[perm,:]
weights=weights[perm]
print("done")

for epoch in xrange(3):
	for i in xrange(5000):
		print(i)
		try:

			tic()
			pos=perturb(samples[i,:])
			region = get_region(pos)
			cutout=RegionCutout(region)
			if (np.max(cutout.get_changed()) > 0):
				raise ReconstructionException("Already changed here")
			toc()

			#check if segment leaves window. If not, don't grow it.
			central_segment = expand_list(cutout.get_subgraph(),[raw_labels[tuple(pos)]])
			central_segment_mask = reconstruct_utils.indicator(cutout.get_raw_labels(),central_segment)
			central_segment_bbox = ndimage.find_objects(central_segment_mask, max_label=1)[0]
			if all([x.stop-x.start < y/3 for x,y in zip(central_segment_bbox,patch_size)]):
				raise ReconstructionException("dust; not growing")
				#print(central_segment_bbox)

			
			tic()
			current_segments = expand_list(cutout.get_subgraph(),[raw_labels[tuple(pos)]]+cutout.local_errors(threshold=0.7))
			toc()

			tic()
			mask_cutout=reconstruct_utils.indicator(cutout.get_raw_labels(),current_segments)
			toc()

			tic()
			t = (reconstruct_utils.flood_fill(cutout.get_image(), mask_cutout), cutout)
			toc()

			"""
			tic()
			analyze(t[0],t[1],i)
			toc()
			"""

			tic()
			commit(*t,low_threshold=0.25,high_threshold=0.75)
			toc()

			"""
			tic()
			central_segment = expand_list(G,[raw_labels[tuple(pos)]])
			central_segment_mask = reconstruct_utils.indicator(cutout.get_raw_labels(),central_segment)
			errors[region]= (1-central_segment_mask)*errors[region] + central_segment_mask * reconstruct_utils.discrim(central_segment_mask)
			toc()
			"""
			print("Committed!")
		except ReconstructionException as e:
			print(e)
			misc_utils.tics=[]
	recompute_errors(epoch=epoch)
	sort_samples()
	
datalist.to_pickle("tmp.pickle")
h5write(os.path.join(basename,"vertices_revised3.h5"),np.array(G.nodes()))
h5write(os.path.join(basename,"edges_revised3.h5"),np.array(G.edges()))
