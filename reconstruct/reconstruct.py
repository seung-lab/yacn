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

PERTURB_RADIUS=(1,15,15)
patch_size=[33,318,318]
full_size=[256,2048,2048]
LOW_THRESHOLD=0.15
HIGH_THRESHOLD=0.85
DUST_THRESHOLD=[x/2 for x in patch_size]
CENTRAL_CROP = 0.33333
VISITED_CROP = 0.33333
ERRORS_CROP = 0.33333
N_EPOCHS = 1
N_STEPS = 8000
GLOBAL_EXPAND = True
ERROR_THRESHOLD=0.5

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
			subregion = crop_region(patch_size, CENTRAL_CROP)
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

	def local_errors(self, threshold):
		subregion = crop_region(patch_size, ERRORS_CROP)
		unique_list = filter(lambda x: x!=0, np.unique(self.raw_labels[subregion]))

		max_error_list = measurements.maximum(self.errors,self.raw_labels, unique_list)
		additional_segments = [unique_list[i] for i in xrange(len(unique_list)) if max_error_list[i]>threshold or max_error_list[i]==0.0]
		additional_segments = filter(lambda x: x != 0 and x not in self.parent.valid, additional_segments)

		return additional_segments

def get_region(pos):
	if not all([patch_size[i]/2 < pos[i] < (full_size[i] - patch_size[i]/2) for i in range(3)]):
		raise ReconstructionException("out of bounds")
	return tuple([slice(pos[i]-patch_size[i]/2,pos[i]+patch_size[i]-patch_size[i]/2) for i in range(3)])

def crop(A, trim):
	return A[crop_region(A.shape, trim)]

def crop_region(patch_size, trim):
	return tuple([slice(
		int(x*trim),
		int(x - x*trim))
		for x in patch_size])

def analyze(cutout,example_id):
	unique_list = cutout.central_unique_list
	args = [cutout.raw_labels, unique_list]
	guess = measurements.mean(cutout.traced, *args)
	truth = measurements.mean(cutout.human_labels[np.unravel_index(np.argmax(cutout.traced),cutout.raw_labels.shape)]==cutout.human_labels, *args)
	volumes = measurements.sum(np.ones_like(cutout.raw_labels), *args)
	histogram_list = list(ndimage.histogram(cutout.traced, 0, 1, 10, *args))


	positive = [unique_list[i] for i in xrange(len(unique_list)) if guess[i] > 0.5]
	negative = [unique_list[i] for i in xrange(len(unique_list)) if guess[i] <= 0.5]


	new_graph = V.G.subgraph(cutout.unique_list).copy()
	regiongraphs.add_clique(new_graph, positive)
	new_obj = indicator(cutout.raw_labels, bfs(new_graph, positive))
	new_errors_cutout = reconstruct_utils.discrim_online_daemon(cutout.image, new_obj)
	old_errors_cutout = cutout.errors * new_obj
	d_error = crop(new_errors_cutout,0.5) - crop(old_errors_cutout,0.5)
	print(np.histogram(d_error, bins=20, range=(-1.0,1.0)))


	return pd.DataFrame([[guess[i],truth[i],volumes[i],args[1][i],histogram_list[i],example_id] for i in xrange(len(args[1]))],columns=["guess","truth","volume","seg_id","histogram","example_id"])

def commit(cutout, low_threshold=LOW_THRESHOLD, high_threshold=HIGH_THRESHOLD, close = lambda x,y: True):
	V=cutout.parent
	#unique_list = cutout.unique_list
	unique_list = cutout.central_unique_list


	traced_list = measurements.mean(cutout.traced, cutout.raw_labels, unique_list)
	
	if not all([x < low_threshold or x > high_threshold for x in traced_list]):
		raise ReconstructionException("not confident")

	#print(zip(traced_list, unique_list))
	positive = [unique_list[i] for i in xrange(len(unique_list)) if traced_list[i]>high_threshold]
	negative = [unique_list[i] for i in xrange(len(unique_list)) if traced_list[i]<low_threshold]

	for l in positive:
		if l in V.valid:
			raise ReconstructionException("blocking merge to valid segment")

	#if 2 in bfs(V.G,positive):
	#	raise ReconstructionException("blocking merge to glia")
	#print(positive)
	#print(negative)


	"""
	original_components = list(nx.connected_components(V.G.subgraph(cutout.unique_list)))
	regiongraphs.add_clique(V.G,positive, guard=close)
	regiongraphs.delete_bipartite(V.G,positive,negative)
	new_components = list(nx.connected_components(V.G.subgraph(cutout.unique_list)))
	changed_list = set(cutout.unique_list) - set.union(*([set([])]+[s for s in original_components if s in new_components]))
	changed_cutout = indicator(cutout.raw_labels,  changed_list)
	V.changed[cutout.region] = np.maximum(V.changed[cutout.region], changed_cutout)
	"""

def perturb(sample, V, radius=PERTURB_RADIUS):
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
	h5write(os.path.join(basename,name+"_changed.h5"),V.changed)

	print("flattening current seg")
	h5write(os.path.join(basename,name+"_machine_labels.h5"),flatten(V.G,V.raw_labels))

	print("preparing to recompute errors")
	pass_errors = np.minimum(V.errors, 1-V.changed)
	pass_visited = 4*(1 - V.changed)

	print("recomputing errors")
	V.errors = reconstruct_utils.unpack(reconstruct_utils.discrim_daemon(*(map(reconstruct_utils.pack,[machine_labels, samples, pass_errors, pass_visited]))))
	
	h5write(os.path.join(basename,name+"_errors.h5"),V.errors)
	V.changed = np.zeros(full_size, dtype=np.uint8)
	V.visited = np.zeros(full_size, dtype=np.uint8)
	print("done")

def sort_samples(V):
	nsamples = V.samples.shape[0]
	weights = V.errors[[V.samples[:,0],V.samples[:,1],V.samples[:,2]]]
	print(np.histogram(weights, bins=20))
	perm = np.argsort(weights)[::-1]
	V.samples=V.samples[perm,:]

if __name__ == "__main__":
	#basename = sys.argv[1]
	basename=os.path.expanduser("~/mydatasets/3_3_1/")

	print("loading files...")
	vertices = h5read(os.path.join(basename, "vertices.h5"), force=True)
	edges = h5read(os.path.join(basename, "edges.h5"), force=True)

	V = Volume(basename,
			{"image": "image.h5",
			 "errors": "errors4.h5",
			 "raw_labels": "raw.h5",
			 "affinities": "aff.h5",
			 "human_labels": "proofread.h5",
			 #"machine_labels": "mean_agg_tr.h5",
			 "changed": np.zeros(full_size, dtype=np.uint8),
			 "visited": np.zeros(full_size, dtype=np.uint8),
			 "valid": set([]),
			 "G": regiongraphs.make_graph(vertices,edges),
			 "samples": h5read(os.path.join(basename, "samples.h5"), force=True),
			 })
	V.errors = V.errors[:]
	#V.valid = set(bfs(V.G, [2]))


	print("done")

	"""
	full_edges = h5read(os.path.join(basename, "full_raw_edges.h5"), force=True)
	full_G = regiongraphs.make_graph(vertices, full_edges)
	close=lambda x,y: full_G.has_edge(x,y)
	"""
	close = lambda x,y: True

	print("sorting samples...")
	sort_samples(V)
	print("done")

	datalist=pd.DataFrame([],columns=[])

	for epoch in xrange(N_EPOCHS):
		for i in xrange(N_STEPS):
			print(i)
			try:

				tic()
				pos=perturb(V.samples[i,:],V)
				region = get_region(pos)
				cutout=SubVolume(V,region)
				if (V.visited[tuple(pos)] > 0):
					raise ReconstructionException("Already visited here")
				toc()

				#check if segment leaves window. If not, don't grow it.
				central_segment = bfs(cutout.G,[V.raw_labels[tuple(pos)]])
				central_segment_mask = indicator(cutout.raw_labels,central_segment)
				central_segment_bbox = ndimage.find_objects(central_segment_mask, max_label=1)[0]
				if all([x.stop-x.start < y for x,y in zip(central_segment_bbox,DUST_THRESHOLD)]):
					raise ReconstructionException("dust; not growing")
				
				tic()
				if GLOBAL_EXPAND:
					g = V.G
				else:
					g = cutout.G
				current_segments = bfs(g,[V.raw_labels[tuple(pos)]]+cutout.local_errors(threshold=ERROR_THRESHOLD))

				for s in current_segments:
					if s in V.valid:
						raise ReconstructionException("segment already valid")
				toc()

				tic()
				mask_cutout=indicator(cutout.raw_labels,current_segments)
				toc()

				tic()
				cutout.traced = reconstruct_utils.trace_daemon(cutout.image, mask_cutout)
				toc()

				tic()
				visited_cutout = indicator(cutout.raw_labels, bfs(V.G, [V.raw_labels[tuple(pos)]]))
				subregion = crop_region(patch_size,VISITED_CROP)
				V.visited[cutout.region][subregion] = np.maximum(V.visited[cutout.region], visited_cutout)[subregion]
				toc()

				tic()
				datalist = datalist.append(analyze(cutout,i))
				toc()

				"""
				tic()
				commit(cutout, close=close)
				toc()
				"""

				print("Committed!")
			except ReconstructionException as e:
				print(e)
				misc_utils.tics=[]
			if i%100 == 0:
				datalist.to_pickle("tmp.pickle")
		#recompute_errors(V,epoch=epoch)
		#sort_samples(V)
