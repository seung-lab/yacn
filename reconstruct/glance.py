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
import os.path

from regiongraphs import *
from reconstruct import *
from misc_utils import *

import scipy.ndimage.measurements as measurements


patch_size=[33,318,318]
resolution=[40,4,4]
full_size=[256,2048,2048]

neuroglancer.server.debug=False
neuroglancer.server.global_server_args['bind_address']='seungworkstation1000.princeton.edu'
neuroglancer.server.global_server_args['bind_port']=80
neuroglancer.server.global_bind_port2=9100
neuroglancer.volume.ENABLE_MESHES=True

def get_focus():
	return map(int, rev(viewer.state['navigation']['pose']['position']['voxelCoordinates']))

def get_selection():
	s=set(map(int,viewer.state['layers']['raw_labels']['segments']))
	return s

def set_selection(segments,append=False):
	segments = map(int, expand_list(V.G, segments))
	if append:
		segments = segments + list(get_selection())
	segments = sorted(list(set(segments)))
		
	viewer.state['layers']['raw_labels']['segments'] = segments
	print(viewer.state['layers']['raw_labels']['segments'])
	viewer.broadcast()

def set_focus(pos):
	global cutout
	cutout = cutout=SubVolume(V,get_region(pos))
	cutout.pos=pos
	draw_bbox(pos)
	viewer.state['navigation']['pose']['position']['voxelCoordinates'] = rev(pos)
	viewer.broadcast()

def rev(x):
	if type(x) == tuple:
		return tuple(reversed(x))
	else:
		return list(reversed(x))

counter=0

def trace():
	s=get_selection()
	mask = indicator(cutout.raw_labels,s)
	cutout.traced = reconstruct_utils.trace_daemon(cutout.image, mask)

	global counter
	counter += 1
	viewer.add(data=cutout.traced, volume_type='image', name='trace'+str(counter), voxel_size=rev(resolution), offset=[(cutout.pos[i]-patch_size[i]/2)*resolution[i] for i in xrange(3)])
	l=viewer.layers[-1]
	viewer.register_volume(l.volume)
	viewer.state['layers']['trace'+str(counter)]=l.get_layer_spec(viewer.get_server_url())
	viewer.broadcast()
	print("done")

def draw_bbox(position):
	position = rev(position)
	rps = rev(patch_size)
	tmp=[]
	for i in [-1,1]:
		for j in [-1,1]:
			for k in [-1,1]:
				if i == - 1:
					tmp.append(map(operator.add,position, [i*rps[0]/2, j*rps[1]/2, k*rps[2]/2]))
					tmp.append(map(operator.add,position, [-i*rps[0]/2, j*rps[1]/2, k*rps[2]/2]))
				if j == - 1:
					tmp.append(map(operator.add,position, [i*rps[0]/2, j*rps[1]/2, k*rps[2]/2]))
					tmp.append(map(operator.add,position, [i*rps[0]/2, -j*rps[1]/2, k*rps[2]/2]))
				if k == - 1:
					tmp.append(map(operator.add,position, [i*rps[0]/2, j*rps[1]/2, k*rps[2]/2]))
					tmp.append(map(operator.add,position, [i*rps[0]/2, j*rps[1]/2, -k*rps[2]/2]))

	viewer.state['layers']['bbox']['points'] = tmp
	viewer.broadcast()

def select_neighbours(threshold=0.5):
	set_selection(cutout.local_errors(threshold=threshold), append=True)

def perturb_position(radius=(1,15,15)):
	pos=get_focus()
	new_pos = perturb(get_focus(),V,radius=radius)
	set_focus(new_pos)

def load(ind=None,append=False):
	if ind is None:
		pos = get_focus()
		z,y,x = pos
	elif type(ind)==int:
		global current_index
		current_index=ind
		z,y,x = V.samples[ind,:]
		pos = [z,y,x]
	else:
		pos=ind
		z,y,x=pos

	set_selection([int(V.raw_labels[z,y,x])], append=append)
	set_focus(pos)


def auto_trace():
	perturb_position()
	raw_input("press enter to continue")
	select_neighbours(threshold=0.5)
	raw_input("press enter to continue")
	trace()
	raw_input("press enter to continue")
	commit(cutout)
	raw_input("press enter to continue")
	load(cutout.pos)

current_index = 0
def next_index(jump=1):
	global current_index
	current_index = current_index + jump
	return current_index

neuroglancer.set_static_content_source(url='http://seungworkstation1000.princeton.edu:8080')

#basename = sys.argv[1]
basename=os.path.expanduser("~/mydatasets/3_3_1/")
print("loading files...")
vertices = h5read(os.path.join(basename, "vertices.h5"), force=True)
edges = h5read(os.path.join(basename, "edges.h5"), force=True)

V = Volume(basename,
		{"image": "image.h5",
		 "errors": "errors3.h5",
		 "raw_labels": "raw.h5",
		 "affinities": "aff.h5",
		 "machine_labels": "mean_agg_tr.h5",
		 "human_labels": "proofread.h5",
		 "changed": np.zeros(full_size, dtype=np.int32),
		 "valid": set([]),
		 "G": regiongraphs.make_graph(vertices,edges),
		 "samples": h5read(os.path.join(basename, "samples.h5"), force=True),
		 "ds_samples": h5read(os.path.join(basename, "ds/samples.h5"), force=True)
		 })
V.errors = V.errors[:]

print("done")

#graph_server_url=graph_server.start_server(V.G)
#graph_server_url="http://localhost:8088"

print("sorting samples...")
sort_samples(V)
print("...done")

viewer = neuroglancer.Viewer()
def on_state_changed(state):
	if neuroglancer.server.debug:
		print(state)

viewer.on_state_changed = on_state_changed
#viewer.add(data=np.array([[[0]]],dtype=np.uint8), volume_type='image', name='dummy', voxel_size=rev(resolution))
viewer.add(data=V.image, volume_type='image', name='image', voxel_size=rev(resolution))
viewer.add(data=V.errors, volume_type='image', name='errors', voxel_size=rev(resolution))
#viewer.add(data=machine_labels, volume_type='segmentation', name='machine_labels', voxel_size=rev(resolution))
viewer.add(data=V.raw_labels, volume_type='segmentation', name='raw_labels', voxel_size=rev(resolution))
#viewer.add(data=human_labels, volume_type='segmentation', name='human_labels', voxel_size=rev(resolution))
viewer.add(data=[], volume_type='synapse', name="bbox")
#viewer.add(data=np.array([[[0]]],dtype=np.uint8), volume_type='image', name='dummy', voxel_size=rev(resolution))

print('open your browser at:')
print(viewer.__str__())
#webbrowser.open(viewer.__str__())
