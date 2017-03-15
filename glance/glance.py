#!/usr/bin/python

from __future__ import print_function
import graph_server
import os
import sys
import webbrowser
import subprocess
import operator
import h5py

import neuroglancer
import numpy as np
import pandas as pd
from sets import Set
from collections import  defaultdict
import os.path

from multiprocessing import Process, Queue
import glance_utils
import regiongraphs
from regiongraphs import *

import scipy.ndimage.measurements as measurements


patch_size=[318,318,33]
resolution=[4,4,40]
full_size=[2048,2048,256]
#discrim_daemon = glance_utils.ComputeDaemon(glance_utils.run_discrim)
trace_daemon = glance_utils.ComputeDaemon(glance_utils.run_trace)
#error_daemon = glance_utils.ComputeDaemon(glance_utils.run_local_error)

neuroglancer.server.debug=False
neuroglancer.server.global_server_args['bind_port']=3389
neuroglancer.server.global_bind_port2=9100
neuroglancer.volume.ENABLE_MESHES=False

files = []

def h5read(filename):
	try:
		f=h5py.File(filename,'r')
		arr = f['main']
		files.append(f)
		return arr
	except IOError:
		print(filename+' not found')

def get_current_region():
	pos = map(int,viewer.state['navigation']['pose']['position']['voxelCoordinates'])
	return pos, get_region(pos)

def get_region(pos):
	assert all([patch_size[i]/2 < pos[i] < (full_size[i] - patch_size[i]/2) for i in range(3)])
	return tuple([slice(pos[i]-patch_size[i]/2,pos[i]+patch_size[i]-patch_size[i]/2) for i in range(3)])

def get_selected_set():
	s=Set(map(int,viewer.state['layers']['raw_labels']['segments']))
	return s

def rev_tuple(x):
	return tuple(reversed(x))

counter=0

def trace():
	pos,region = get_current_region()
	s=get_selected_set()
	labels_cutout = np.copy(raw_labels[rev_tuple(region)])
	image_cutout = np.copy(image[rev_tuple(region)])
	mask=glance_utils.indicator(labels_cutout,s)
	traced = trace_daemon(image_cutout, mask)

	global counter
	counter += 1
	viewer.add(data=traced, volume_type='image', name='trace'+str(counter), voxel_size=resolution, offset=[(pos[i]-patch_size[i]/2)*resolution[i] for i in xrange(3)])
	l=viewer.layers[-1]
	viewer.register_volume(l.volume)
	viewer.state['layers']['trace'+str(counter)]=l.get_layer_spec(viewer.get_server_url())
	viewer.broadcast()
	print("done")
	return traced, labels_cutout

def commit(traced, labels_cutout, low_threshold=0.3, high_threshold=0.7):
	unique_list = np.unique(labels_cutout)
	traced_list = measurements.mean(traced, labels_cutout, unique_list)
	print(zip(traced_list, unique_list))
	positive = [unique_list[i] for i in xrange(len(unique_list)) if traced_list[i]>high_threshold]
	negative = [unique_list[i] for i in xrange(len(unique_list)) if traced_list[i]<low_threshold]
	positive = filter(lambda x: x != 0, positive)
	negative = filter(lambda x: x != 0, negative)
	print(positive)
	print(negative)
	regiongraphs.add_clique(G,positive)
	regiongraphs.delete_bipartite(G,positive,negative)


def discrim():
	pos, region = get_current_region()
	s=get_selected_set()
	labels_cutout = np.copy(raw_labels[rev_tuple(region)])
	image_cutout = np.copy(image[rev_tuple(region)])
	mask = glance_utils.indicator(labels_cutout,s)
	
	return discrim_daemon(mask)

def draw_bbox(current_position):
	tmp=[]
	for i in [-1,1]:
		for j in [-1,1]:
			for k in [-1,1]:
				if i == - 1:
					tmp.append(map(operator.add,current_position, [i*patch_size[0]/2, j*patch_size[1]/2, k*patch_size[2]/2]))
					tmp.append(map(operator.add,current_position, [-i*patch_size[0]/2, j*patch_size[1]/2, k*patch_size[2]/2]))
				if j == - 1:
					tmp.append(map(operator.add,current_position, [i*patch_size[0]/2, j*patch_size[1]/2, k*patch_size[2]/2]))
					tmp.append(map(operator.add,current_position, [i*patch_size[0]/2, -j*patch_size[1]/2, k*patch_size[2]/2]))
				if k == - 1:
					tmp.append(map(operator.add,current_position, [i*patch_size[0]/2, j*patch_size[1]/2, k*patch_size[2]/2]))
					tmp.append(map(operator.add,current_position, [i*patch_size[0]/2, j*patch_size[1]/2, -k*patch_size[2]/2]))

	viewer.state['layers']['bbox']['points'] = tmp
	viewer.broadcast()

def load_neighbours(threshold=0.1):
	pos, region = get_current_region()
	x,y,z=pos
	current_segments = [int(raw_labels[z,y,x])]

	labels_cutout = np.copy(raw_labels[rev_tuple(region)])
	errors_cutout = np.copy(errors[rev_tuple(region)])


	unique_list = np.unique(labels_cutout)
	max_error_list = measurements.maximum(errors_cutout,labels_cutout, unique_list)
	additional_segments = [unique_list[i] for i in xrange(len(unique_list)) if max_error_list[i]>threshold]

	print(current_segments + additional_segments)

	viewer.state['layers']['raw_labels']['segments'] = sorted(map(int,expand_list(G,current_segments + additional_segments)))
	viewer.broadcast()

def perturb(sample,radius):
	#sample should be in xyz order
	region = tuple([slice(x-y,x+y+1,None) for x,y in zip(sample,radius)])
	print(region)
	mask = (raw_labels[rev_tuple(region)]==raw_labels[rev_tuple(tuple(sample))]).astype(np.int32)

	patch = np.minimum(affinities[(0,)+rev_tuple(region)], mask)
	tmp=np.unravel_index(patch.argmax(),patch.shape)
	return [t+x-y for t,x,y in zip(reversed(tmp),sample,radius)]

def perturb_position(radius=(15,15,1)):
	pos,current_region=get_current_region()
	set_location(perturb(pos,radius=radius))

def load(ind=None):
	if ind is None:
		pos,region = get_current_region()
		x,y,z=pos
	elif type(ind)==int:
		z,y,x = samples[ind,:]
		pos = [x,y,z]
	else:
		x,y,z=ind
	current_segments = [int(raw_labels[z,y,x])]
	set_selection(current_segments)
	set_location(pos)
	draw_bbox(pos)

def set_selection(segments):
	viewer.state['layers']['raw_labels']['segments'] = sorted(map(int,expand_list(G,segments)))
	print(viewer.state['layers']['raw_labels']['segments'])
	viewer.broadcast()

def set_location(pos):
	viewer.state['navigation']['pose']['position']['voxelCoordinates'] = pos
	viewer.broadcast()


neuroglancer.set_static_content_source(url='http://seungworkstation15.princeton.edu:8080')

#basename = sys.argv[1]
basename=os.path.expanduser("~/mydatasets/3_3_1/")
print("loading files...")
with h5py.File(os.path.join(basename,"samples.h5"),'r') as f:
	#careful! These are in z,y,x order
	samples = f['main'][:]

with h5py.File(os.path.join(basename,"vertices.h5"),'r') as f:
	vertices = f['main'][:]

with h5py.File(os.path.join(basename,"edges.h5"),'r') as f:
	edges= f['main'][:]

image = h5read(os.path.join(basename,"image.h5"))
errors = h5read(os.path.join(basename,"errors3.h5"))
raw_labels = h5read(os.path.join(basename,"raw.h5"))
affinities = h5read(os.path.join(basename,"aff.h5"))
#human_labels = h5read(os.path.join(basename,"proofread.h5"))
#machine_labels= h5read(os.path.join(basename,"mean_agg_tr.h5"))


G=regiongraphs.make_graph(vertices,edges)
#graph_server_url=graph_server.start_server(G)
#graph_server_url="http://localhost:8088"

errors=errors[:]
print("...done")

print("sorting samples...")
nsamples = samples.shape[0]
weights = errors[[samples[:,0],samples[:,1],samples[:,2]]]

perm = np.argsort(weights)[::-1]
samples=samples[perm,:]
weights=weights[perm]
print("...done")

viewer = neuroglancer.Viewer()
def on_state_changed(state):
	if neuroglancer.server.debug:
		print(state)

viewer.on_state_changed = on_state_changed
#viewer.add(data=np.array([[[0]]],dtype=np.uint8), volume_type='image', name='dummy', voxel_size=resolution)
viewer.add(data=image, volume_type='image', name='image', voxel_size=resolution)
viewer.add(data=errors, volume_type='image', name='errors', voxel_size=resolution)
#viewer.add(data=machine_labels, volume_type='segmentation', name='machine_labels', voxel_size=resolution)
viewer.add(data=raw_labels, volume_type='segmentation', name='raw_labels', voxel_size=resolution)
#viewer.add(data=human_labels, volume_type='segmentation', name='human_labels', voxel_size=resolution)
viewer.add(data=[], volume_type='synapse', name="bbox")
#viewer.add(data=np.array([[[0]]],dtype=np.uint8), volume_type='image', name='dummy', voxel_size=resolution)

print('open your browser at:')
print(viewer.__str__())
#webbrowser.open(viewer.__str__())
