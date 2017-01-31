import webbrowser

import neuroglancer
import numpy as np
import time
import threading
from select import select
import sys

neuroglancer.set_static_content_source(url="https://neuroglancer-demo.appspot.com")
viewer = neuroglancer.Viewer(voxel_size=[6,6,30])
viewer.add(np.zeros([48,314,314],dtype=np.float32))

def get_bool():
	while True:
		rlist, _, _ = select([sys.stdin], [], [], 1)
		if rlist:
			s = sys.stdin.readline()
			return int(s)==1
		else:
			time.sleep(0)

cnt=0
def validate(A):
	global viewer
	viewer.add(A.astype(np.uint32), name=str(cnt))
	print viewer.__str__()
	x=get_bool()
	print x
	return x
	
	
