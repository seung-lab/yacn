import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from experiments import save_experiment, repo_root
from interrupt_handler import DelayedKeyboardInterrupt
from tensorflow.python.client import timeline
import time

dtype=tf.float32
shape_dict3d={}
shape_dict2d={}
shape_dictz={}

class Model():

	def restore(self, modelpath):
		modelpath = os.path.expanduser(modelpath)
		self.saver.restore(self.sess, modelpath)

	def train(self, nsteps=100000, checkpoint_interval=1000):
		self.init_log()
		print ('log initialized')
		for i in xrange(nsteps):
			try:
				with DelayedKeyboardInterrupt():
					t = time.time()
					if i==5:
						_, quick_summary = self.sess.run(
							[self.train_op, self.quick_summary_op], options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=self.run_metadata)
						trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)
						trace_file = open('timeline.ctf.json', 'w')
						trace_file.write(trace.generate_chrome_trace_format())
					else:
						_, quick_summary = self.sess.run(
							[self.train_op, self.quick_summary_op])

					elapsed = time.time() - t
					print("elapsed: ", elapsed)
					self.summary_writer.add_summary(
						quick_summary, self.sess.run(self.step))
					if i % checkpoint_interval == 0:
						print("checkpointing...")
						step = self.sess.run(self.step)
						self.saver.save(self.sess, self.logdir + "model" + str(step) + ".ckpt")
						self.summary_writer.add_summary(self.sess.run(self.summary_op), step)
						self.summary_writer.flush()
						print("done")
			except KeyboardInterrupt:
				break
				#self.interrupt()

class Volume():
	def __init__(self, A, patch_size):
		self.A=A
		self.patch_size=patch_size

	def __getitem__(self, focus):
		A=self.A
		patch_size=self.patch_size
		with tf.device("/cpu:0"):
			corner = focus - np.array([x/2 for x in patch_size],dtype=np.int32)
			corner = tf.unpack(corner)
			return tf.stop_gradient(tf.slice(A,corner,patch_size))

	def __setitem__(A, focus, vol):
		A=self.A
		patch_size=self.patch_size
		with tf.device("/cpu:0"):
			corner = focus - np.array([x/2 for x in patch_size],dtype=np.int32)
			corner = tf.Print(corner,[corner])
			corner = tf.unpack(corner)
			return tf.stop_gradient(A[corner[0]:corner[0]+patch_size[0],
						corner[1]:corner[1]+patch_size[1],
						corner[2]:corner[2]+patch_size[2]].assign(vol))

def random_row(A):
	index=tf.random_uniform([],minval=0,maxval=static_shape(A)[0],dtype=tf.int32)
	return A[index,:]
def unique(x):
	tmp = tf.reshape(x, [-1])
	tmp = tf.unique(tmp)[1]
	return tf.reshape(tmp, static_shape(x))

def KL(a,b):
	return -a*tf.log(b/a)-(1-a)*tf.log((1-b)/(1-a))

def local_error(ground_truth, proposal):
	T=tf.reduce_sum(ground_truth)
	P=tf.reduce_sum(proposal)
	S=tf.reduce_sum(proposal*ground_truth)
	return [T, P, S]

def random_occlusion(target):
	target = tf.to_float(tf_pad_shape(target))
	patch_size = static_shape(target)

	xmask=tf.to_float(tf.concat(0,[tf.ones((patch_size[0]/2,patch_size[1],patch_size[2],1)),tf.zeros((patch_size[0]/2,patch_size[1],patch_size[2],1))]))
	ymask=tf.to_float(tf.concat(1,[tf.ones((patch_size[0],patch_size[1]/2,patch_size[2],1)),tf.zeros((patch_size[0],patch_size[1]/2,patch_size[2],1))]))
	zmask=tf.to_float(tf.concat(2,[tf.ones((patch_size[0],patch_size[1],patch_size[2]/2,1)),tf.zeros((patch_size[0],patch_size[1],patch_size[2]/2,1))]))
	full = tf.to_float(tf.ones(patch_size))

	xmasks = tf.pack([xmask, 1-xmask, full])
	ymasks = tf.pack([ymask, 1-ymask, full])
	zmasks = tf.pack([zmask, 1-ymask, full])

	xchoice = tf.reshape(tf.one_hot(tf.multinomial([0.3*tf.log(0.001+tf.reduce_sum(xmasks*tf.pack([target]), reduction_indices=[1,2,3,4]))],1),3),(3,1,1,1,1))
	ychoice = tf.reshape(tf.one_hot(tf.multinomial([0.3*tf.log(0.001+tf.reduce_sum(ymasks*tf.pack([target]), reduction_indices=[1,2,3,4]))],1),3),(3,1,1,1,1))
	zchoice = tf.reshape(tf.one_hot(tf.multinomial([0.3*tf.log(0.001+tf.reduce_sum(zmasks*tf.pack([target]), reduction_indices=[1,2,3,4]))],1),3),(3,1,1,1,1))

	mask = tf.reduce_sum(xmasks*xchoice, reduction_indices=0) * \
	tf.reduce_sum(ymasks*ychoice, reduction_indices=0) * \
	tf.reduce_sum(zmasks*zchoice, reduction_indices=0)
	return mask*target

def trimmed_sigmoid(logit):
	return 0.00001+0.99998*tf.nn.sigmoid(logit)

def static_constant_variable(x, fd):
	placeholder = tf.placeholder(tf.as_dtype(x.dtype), shape=x.shape)
	fd[placeholder]=x
	return tf.Variable(placeholder)

def bump_map(patch_size):
	tmp=np.zeros(patch_size)
	I,J,K=tmp.shape
	for i in xrange(I):
		for j in xrange(J):
			for k in xrange(K):
				tmp[i,j,k]=3*bump_logit((i+1.0)/(I+2.0),(j+1.0)/(J+2.0),(k+1.0)/(K+2.0))
	tmp-=np.max(tmp)
	return np.exp(tmp)

def bump_logit(x,y,z):
	t=1
	return -(x*(1-x))**(-t)-(y*(1-y))**(-t)-(z*(1-z))**(-t)

def rand_bool(shape, prob=0.5):
	return tf.less(tf.random_uniform(shape),prob)

def random_rotation(x):
	perm = tf.cond(rand_bool([]), lambda: tf.constant([0,1,2]), lambda: tf.constant([0,2,1]))
	return tf.reshape(tf.transpose(tf.reverse(x, rand_bool([3])), perm=perm),static_shape(x))

def bounded_cross_entropy(guess,truth):
	guess = 0.999998*guess + 0.000001
	return  - truth * tf.log(guess) - (1-truth) * tf.log(1-guess)

def lrelu(x):
	return tf.nn.relu(x) - tf.log(-tf.minimum(x,0)+1)

def prelu(x,n):
	return tf.nn.relu(x) - n*tf.pow((tf.nn.relu(-x)+1),1.0/n) + n

def is_valid(nfeatures,i,j):
	return (i <= j and i >= 0 and j >= 0 and i < len(nfeatures) and j < len(nfeatures[i]) and nfeatures[i][j] > 0)

def conditional_map(fun,lst,cond,default):
	return [tf.cond(c, lambda: fun(l), lambda: default) for l,c in zip(lst, cond)]

def covariance(population, mean, weight):
	population = population - tf.reshape(mean,[1,-1])
	return matmul(tf.transpose(population), population)/weight

def get_pair(A,offset, patch_size):
	os1 = map(lambda x: max(0,x) ,offset)
	os2 = map(lambda x: max(0,-x),offset)
	
	A1 = A[os1[0]:patch_size[0]-os2[0],
		os1[1]:patch_size[1]-os2[1],
		os1[2]:patch_size[2]-os2[2],
		:]
	A2 = A[os2[0]:patch_size[0]-os1[0],
		os2[1]:patch_size[1]-os1[1],
		os2[2]:patch_size[2]-os1[2],
		:]
	return (A1, A2)

def label_diff(x,y):
	return tf.to_float(tf.equal(x,y))

def norm(A):
	return tf.reduce_sum(tf.square(A),reduction_indices=[3],keep_dims=True)

def block_cat(A,B,C,D):
	n=len(A.get_shape())
	return tf.concat(n-2,[tf.concat(n-1,[A,B]),tf.concat(n-1,[C,D])])

def vec_cat(A,B):
	n=len(A.get_shape())
	return tf.concat(n-2,[A,B])

def matmul(*l):
	return reduce(tf.matmul,l)

def batch_matmul(*l):
	return reduce(tf.batch_matmul,l)

def batch_transpose(A):
	n=len(A.get_shape())
	perm = range(n-2)+[n-1,n-2]
	return tf.transpose(A,perm=perm)

def categorical(logits, selection_logit=False):
	s=static_shape(logits)
	logits = tf.reshape(logits,[-1])
	U=tf.random_uniform(logits.get_shape())
	x=tf.to_int32(tf.argmax(logits - tf.log(-tf.log(U)),0))
	ret = linear_to_ind(x,s)
	if selection_logit:
		return ret, logits[x]
	else:
		return ret

def vector_argmax(A):
	n=len(static_shape(A))
	if n == 0:
		assert False
	elif n == 1:
		return [tf.argmax(A,0)]
	else:
		#first we need to locate which subarray contains the maximum
		maxval = tf.argmax(tf.reduce_max(A,reduction_indices=range(1,n)),0)
		return [maxval] + vector_argmax(tf.squeeze(tf.slice(A,[maxval]+[0]*(n-1),[1]+static_shape(A)[1:])))

def categorical2(logits):
	s=static_shape(logits)
	U=tf.random_uniform(logits.get_shape())
	return tf.to_int32(tf.pack(vector_argmax(logits - tf.log(-tf.log(U)))))
	
def linear_to_ind(a,shape):
	t=[]
	for i in reversed(shape):
		t.insert(0,tf.mod(a,i))
		a=tf.floordiv(a,i)
	
	#should t be reversed?
	return t

def logdet(M):
	return tf.reduce_sum(tf.log(tf.self_adjoint_eig(M)[0,:]))

def collapse_image(x):
	return tf.expand_dims(tf.concat(1, tf.unpack(x)),0)

def image_summary(name, x):
	return tf.image_summary(name,tf.transpose(collapse_image(x), perm=[3,1,2,0]), max_images=6)

def image_slice_summary(name, x):
	return tf.image_summary(name,x[16:17,:,:,:], max_images=6)

def image_summary_pad(name,x):
	return image_summary(name,tf_pad_shape(x))

def colour_image_summary(name, x):
	return tf.image_summary(name,collapse_image(x), max_images=6)

def pad_shape(A):
	return np.reshape(A, list(np.shape(A)) + [1])

def tf_pad_shape(A):
	return tf.reshape(A,list(static_shape(A))+[1])

def constant_variable(shape, val=0.0):
	initial = tf.constant(val, dtype=dtype, shape=shape)
	var = tf.Variable(initial, dtype=dtype)
	return var

def normal_variable(shape, stddev=0.1):
	initial = tf.truncated_normal(shape, dtype=dtype, stddev=stddev)
	return tf.Variable(initial, dtype=dtype)

def identity_matrix(n):
	return tf.diag(tf.ones([n]))

def static_shape(x):
	return [x.value for x in x.get_shape()]