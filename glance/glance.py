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
#discrim_daemon = glance_utils.ComputeDaemon(glance_utils.run_discrim)
trace_daemon = glance_utils.ComputeDaemon(glance_utils.run_trace)
#error_daemon = glance_utils.ComputeDaemon(glance_utils.run_local_error)

neuroglancer.server.debug=False
neuroglancer.server.global_server_args['bind_port']=80
neuroglancer.server.global_bind_port2=9100

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

def load_example(example_ind=None):
	if example_ind is None:
		pos,region = get_current_region()
		x,y,z=pos
	else:
		z,y,x = samples[example_ind,:]
		pos = [x,y,z]
	current_segments = [int(raw_labels[z,y,x])]

	viewer.state['layers']['raw_labels']['segments'] = sorted(map(int,expand_list(G,current_segments)))
	print(viewer.state['layers']['raw_labels']['segments'])
	viewer.state['navigation']['pose']['position']['voxelCoordinates'] = pos
	draw_bbox(pos)
	viewer.broadcast()

neuroglancer.set_static_content_source(url='http://seungworkstation15.princeton.edu:8080')

h5_file_handles=[]
def h5read(path):
	f=h5py.File(path,'r')
	h5_file_handles.append(f)
	return f['main']

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
human_labels = h5read(os.path.join(basename,"proofread.h5"))


G=regiongraphs.make_graph(vertices,edges)
#graph_server_url=graph_server.start_server(G)
#graph_server_url="http://localhost:8088"

print("...done")

print("sorted samples...")
nsamples = samples.shape[0]
errors=errors[:]
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
#viewer.add(data=labels, volume_type='segmentation', name='labels', voxel_size=resolution)
viewer.add(data=raw_labels, volume_type='segmentation', name='raw_labels', voxel_size=resolution)
#viewer.add(data=human_labels, volume_type='segmentation', name='human_labels', voxel_size=resolution)
viewer.add(data=[], volume_type='synapse', name="bbox")
#viewer.add(data=np.array([[[0]]],dtype=np.uint8), volume_type='image', name='dummy', voxel_size=resolution)

print('open your browser at:')
print(viewer.__str__())
#webbrowser.open(viewer.__str__())
