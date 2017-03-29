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

class Volume():
	def __init__(self, directory, d):
		self.directory=directory
		for (k,v) in d.items():
			if type(v)==type(""):
				setattr(self, k, h5read(os.path.join(directory, v)))
			else:
				setattr(self, k, v)

class SubVolume():
	def __init__(self, parent, region):
		self.parent = parent
		self.region = region
	
	def __getattr__(self, name):
		if name == "unique_list":
			self.unique_list = filter(lambda x: x!=0, np.unique(self.raw_labels))
		elif name == "central_unique_list":
			subregion = tuple([slice(x/3,x-x/3) for x in patch_size])
			self.central_unique_list = filter(lambda x: x!=0, np.unique(self.raw_labels[subregion]))
		elif name == "G":
			self.G = self.parent.G.subgraph(self.unique_list)
		elif name == "local_labels":
			tic()
			components=nx.connected_components(self.G)
			d={}
			for i,nodes in enumerate(components,1):
				for node in nodes:
					d[node]=i
			d[0]=0
			self.local_labels = np.vectorize(d.get)(self.raw_labels)
			toc("local labels")
		else:
			setattr(self, name, getattr(self.parent,name)[self.region])
		return getattr(self,name)

	def local_errors(self, threshold=0.5):
		unique_list = self.unique_list
		#unique_list = self.get_central_unique_list()
		max_error_list = measurements.maximum(self.errors,self.raw_labels, unique_list)
		additional_segments = [unique_list[i] for i in xrange(len(unique_list)) if max_error_list[i]>threshold or max_error_list[i]==0.0]
		additional_segments = filter(lambda x: x != 0, additional_segments)

		return additional_segments

def get_region(pos):
	if not all([patch_size[i]/2 < pos[i] < (full_size[i] - patch_size[i]/2) for i in range(3)]):
		raise ReconstructionException("out of bounds")
	return tuple([slice(pos[i]-patch_size[i]/2,pos[i]+patch_size[i]-patch_size[i]/2) for i in range(3)])

"""
datalist=pd.DataFrame([],columns=["guess","truth","volume","seg_id","example_id"])

def analyze(traced, cutout,example_id):
	args = [cutout.local_labels, filter(lambda x: x!=0, np.unique(cutout.local_labels))]
	guess = measurements.mean(traced, *args)
	truth = measurements.mean(cutout.human_labels[np.unravel_index(np.argmax(traced),cutout.raw_labels.shape)]==cutout.human_labels, *args)
	volumes = measurements.sum(np.ones_like(cutout.raw_labels), *args)

	global datalist
	datalist=datalist.append(pd.DataFrame([[guess[i],truth[i],volumes[i],args[1][i],example_id] for i in xrange(len(args[1]))],columns=["guess","truth","volume","seg_id","example_id"]))
"""

def commit(traced, cutout, low_threshold=0.2, high_threshold=0.8):
	V=cutout.parent
	#unique_list = cutout.unique_list
	unique_list = cutout.central_unique_list


	traced_list = measurements.mean(traced, cutout.raw_labels, unique_list)
	
	if not all([x < low_threshold or x > high_threshold for x in traced_list]):
		raise ReconstructionException("not confident")

	#print(zip(traced_list, unique_list))
	positive = [unique_list[i] for i in xrange(len(unique_list)) if traced_list[i]>high_threshold]
	negative = [unique_list[i] for i in xrange(len(unique_list)) if traced_list[i]<low_threshold]

	for l in positive:
		if l in V.valid:
			raise ReconstructionException("blocking merge to valid segment")

	if 2 in expand_list(V.G,positive):
		raise ReconstructionException("blocking merge to glia")
	#print(positive)
	#print(negative)
	original_components = list(nx.connected_components(V.G.subgraph(unique_list)))
	global close
	regiongraphs.add_clique(V.G,positive, guard=close)
	regiongraphs.delete_bipartite(V.G,positive,negative)
	new_components = list(nx.connected_components(V.G.subgraph(unique_list)))
	changed_list = set(unique_list) - set.union(*([set([])]+[s for s in original_components if s in new_components]))
	changed_cutout = indicator(cutout.raw_labels,  changed_list)
	V.changed[cutout.region] = np.maximum(V.changed[cutout.region], changed_cutout)

def perturb(sample, V, radius=(1,15,15)):
	region = tuple([slice(x-y,x+y+1,None) for x,y in zip(sample,radius)])
	mask = (V.raw_labels[region]==V.raw_labels[tuple(sample)]).astype(np.int32)

	patch = np.minimum(V.affinities[(0,)+region], mask)
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

def recompute_errors(V, epoch=None):
	if epoch is None:
		name = "epoch"
	else:
		name = "epoch" + str(epoch)
	h5write(os.path.join(basename,name+"_vertices.h5"),np.array(V.G.nodes()))
	h5write(os.path.join(basename,name+"_edges.h5"),np.array(V.G.edges()))
	h5write(os.path.join(basename,name+"_changed.h5"),V.changed[:,::2,::2])

	print("flattening current seg")
	sub_raw_labels = V.raw_labels[:,::2,::2]
	sub_machine_labels = flatten(V.G, sub_raw_labels)
	h5write(os.path.join(basename,name+"_sub_machine_labels.h5"),sub_machine_labels)
	h5write(os.path.join(basename,name+"_machine_labels.h5"),flatten(V.G,V.raw_labels))

	print("preparing to recompute errors")
	sub_errors = np.minimum(errors[:,::2,::2], 1-V.changed[:,::2,::2])
	sub_visited = 4*(1 - V.changed[:, ::2, ::2])
	sub_samples = np.array(filter(lambda i: sub_visited[i[0],i[1],i[2]]==0, ds_samples))
	print(sub_samples.shape)

	print("recomputing errors")
	sub_new_errors = glance_utils.unpack(reconstruct_utils.discrim_daemon(*(map(glance_utils.pack,[sub_machine_labels, sub_samples, sub_errors, sub_visited]))))
	
	V.errors = np.zeros_like(errors)
	V.errors[:,::2,::2] = sub_new_errors

	h5write(os.path.join(basename,name+"_errors.h5"),sub_new_errors)
	V.changed = np.zeros_like(V.machine_labels, dtype=np.int32)
	print("done")

def sort_samples():
	global nsamples
	global samples
	nsamples = samples.shape[0]
	weights = V.errors[[samples[:,0],samples[:,1],samples[:,2]]]
	perm = np.argsort(weights)[::-1]
	samples=samples[perm,:]

patch_size=[33,318,318]
full_size=[256,2048,2048]


if __name__ == "__main__":
	#basename = sys.argv[1]
	basename=os.path.expanduser("~/mydatasets/3_3_1/")

	print("loading files...")
	vertices = h5read(os.path.join(basename, "vertices.h5"), force=True)
	edges = h5read(os.path.join(basename, "edges.h5"), force=True)
	ds_samples = h5read(os.path.join(basename, "ds/samples.h5"), force=True)
	samples = h5read(os.path.join(basename, "samples.h5"), force=True)

	V = Volume(basename,
			{"image": "image.h5",
			 "errors": "errors4.h5",
			 "raw_labels": "raw.h5",
			 "affinities": "aff.h5",
			 "machine_labels": "mean_agg_tr.h5",
			 "changed": np.zeros(full_size, dtype=np.int32),
			 "valid": set([]),
			 "G": regiongraphs.make_graph(vertices,edges)
			 })
	V.errors = V.errors[:]

	print("done")

	#close=misc_utils.compute_fullgraph(raw_labels[:,::2,::2], resolution=[8,8,40], r=100)
	close = lambda x,y: True

	print("sorting samples...")
	sort_samples()
	print("done")

	for epoch in xrange(2):
		for i in xrange(8000):
			print(i)
			try:

				tic()
				pos=perturb(samples[i,:],V)
				region = get_region(pos)
				cutout=SubVolume(V,region)
				#if (np.max(cutout.changed) > 0):
				if (V.changed[tuple(pos)] > 0):
					raise ReconstructionException("Already changed here")
				toc()

				#check if segment leaves window. If not, don't grow it.
				central_segment = expand_list(cutout.G,[V.raw_labels[tuple(pos)]])
				central_segment_mask = reconstruct_utils.indicator(cutout.raw_labels,central_segment)
				central_segment_bbox = ndimage.find_objects(central_segment_mask, max_label=1)[0]
				if all([x.stop-x.start < y/3 for x,y in zip(central_segment_bbox,patch_size)]):
					raise ReconstructionException("dust; not growing")
					#print(central_segment_bbox)
				
				tic()
				current_segments = expand_list(cutout.G,[V.raw_labels[tuple(pos)]]+cutout.local_errors(threshold=0.5))
				toc()

				tic()
				mask_cutout=reconstruct_utils.indicator(cutout.raw_labels,current_segments)
				toc()

				tic()
				t = (reconstruct_utils.flood_fill(cutout.image, mask_cutout), cutout)
				toc()

				"""
				tic()
				analyze(t[0],t[1],i)
				toc()
				"""

				tic()
				commit(*t,low_threshold=0.15,high_threshold=0.85)
				toc()

				"""
				tic()
				central_segment = expand_list(G,[raw_labels[tuple(pos)]])
				central_segment_mask = reconstruct_utils.indicator(cutout.raw_labels,central_segment)
				errors[region]= (1-central_segment_mask)*errors[region] + central_segment_mask * reconstruct_utils.discrim(central_segment_mask)
				toc()
				"""
				print("Committed!")
			except ReconstructionException as e:
				print(e)
				misc_utils.tics=[]
		recompute_errors(epoch=epoch)
		sort_samples()
		
	#datalist.to_pickle("tmp.pickle")
