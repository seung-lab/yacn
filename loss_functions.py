from __future__ import print_function
import tensorflow as tf
from utils import *
import math

def bounded_cross_entropy(guess,truth):
	guess = 0.999998*guess + 0.000001
	return  - truth * tf.log(guess) - (1-truth) * tf.log(1-guess)

def label_diff(x,y):
	return tf.to_float(tf.equal(x,y))


def length_scale(x):
	if x == 0:
		return -1
	else:
		return math.log(abs(x))


def valid_pair(x, y, strict=False):
	return x == 0 or y == 0 or (
		(not strict or length_scale(x) >= length_scale(y)) and abs(
			length_scale(x) - length_scale(y)) <= math.log(3.1))


def valid_offset(x):
	return x > (0,0,0) and \
	valid_pair(4 * x[0], x[1], strict=True) and \
	valid_pair(4 * x[0], x[2], strict=True) and \
	valid_pair(x[1],x[2])

def get_pair(A,offset, patch_size):
	os1 = map(lambda x: max(0,x) ,offset)
	os2 = map(lambda x: max(0,-x),offset)
	
	A1 = A[:,os1[0]:patch_size[0]-os2[0],
		os1[1]:patch_size[1]-os2[1],
		os1[2]:patch_size[2]-os2[2],
		:]
	A2 = A[:,os2[0]:patch_size[0]-os1[0],
		os2[1]:patch_size[1]-os1[1],
		os2[2]:patch_size[2]-os1[2],
		:]
	return (A1, A2)

#We will only count edges with at least one endpoint in mask
def long_range_loss_fun(vec_labels, human_labels, offsets, mask):
	patch_size = static_shape(vec_labels)[1:4]
	cost = 0
	otpts = {}


	for i, offset in enumerate(offsets):
		guess = affinity(
			*get_pair(vec_labels,offset,patch_size))
		truth = label_diff(
				*get_pair(human_labels,offset,patch_size))

		curr_mask = tf.maximum(*get_pair(mask, offset, patch_size))

		otpts[offset] = guess

		cost += tf.reduce_sum(curr_mask * bounded_cross_entropy(guess, truth))

	return cost, otpts
def affinity(x, y):
	displacement = x - y
	interaction = tf.reduce_sum(
		displacement * displacement,
		reduction_indices=[4],
		keep_dims=True)
	return tf.exp(-0.5 * interaction)

def batch_interaction(mu0, cov0, mu1, cov1, maxn=40, nvec_labels=20):
	propagator = identity_matrix(nvec_labels)
	inv_propagator = identity_matrix(nvec_labels)


	A = tf.reshape(propagator, [1, 1, nvec_labels, nvec_labels])
	invA = tf.reshape(inv_propagator, [1, 1, nvec_labels, nvec_labels])
	mu0 = tf.reshape(mu0, [maxn, 1, nvec_labels, 1])
	mu1 = tf.reshape(mu1, [1, maxn, nvec_labels, 1])
	cov0 = tf.reshape(cov0, [maxn, 1, nvec_labels, nvec_labels])
	cov1 = tf.reshape(cov1, [1, maxn, nvec_labels, nvec_labels])
	identity = tf.reshape(tf.diag(tf.ones((nvec_labels,))), [
						  1, 1, nvec_labels, nvec_labels])
	identity2 = tf.reshape(identity_matrix(
		2 * nvec_labels), [1, 1, 2 * nvec_labels, 2 * nvec_labels])

	cov0 = cov0 + 0.0001 * identity
	cov1 = cov1 + 0.0001 * identity

	delta = mu1 - mu0
	delta2 = vec_cat(delta, delta)

	sqcov0 = tf.tile(tf.cholesky(cov0), [1, maxn, 1, 1])
	sqcov1 = tf.tile(tf.cholesky(cov1), [maxn, 1, 1, 1])
	invA = tf.tile(invA, [maxn, maxn, 1, 1])
	A = tf.tile(A, [maxn, maxn, 1, 1])

	scale = block_cat(
		sqcov0,
		tf.zeros_like(sqcov0),
		tf.zeros_like(sqcov0),
		sqcov1)
	M = batch_matmul(
		batch_transpose(scale),
		block_cat(
			invA,
			invA,
			invA,
			invA),
		scale) + identity2
	scale2 = block_cat(invA, tf.zeros_like(A), tf.zeros_like(A), invA)

	v = batch_matmul(
		tf.matrix_inverse(M),
		batch_transpose(scale),
		scale2,
		delta2)

	ret1 = 1 / tf.sqrt(tf.matrix_determinant(M))
	ret2 = tf.exp(-0.5 * (batch_matmul(batch_transpose(delta),
				  invA, delta) - batch_matmul(batch_transpose(v), M, v)))

	ret = ret1 * tf.squeeze(ret2, [2, 3])

	return ret



def label_loss_fun(vec_labels, human_labels, central_labels, central_labels_mask, maxn=40):
	nvec_labels = static_shape(vec_labels)[4]

	weight_mask = tf.maximum(identity_matrix(maxn),tf.maximum(tf.reshape(central_labels,[-1,1])*tf.ones([1,maxn]),tf.ones([maxn,1])*tf.reshape(central_labels,[1,-1])))


	human_labels = tf.reshape(human_labels, [-1])
	vec_labels = tf.reshape(vec_labels, [-1, nvec_labels])
	central_labels_mask = tf.reshape(central_labels_mask, [-1])
	sums = tf.unsorted_segment_sum(vec_labels, human_labels, maxn)
	weights = tf.to_float(tf.unsorted_segment_sum(tf.ones(static_shape(human_labels)), human_labels, maxn))
	safe_weights = tf.maximum(weights, 0.1)
	
	means = sums / tf.reshape(safe_weights, [maxn,1])
	centred_vec_labels = vec_labels - tf.gather(means, human_labels)
	full_covs = tf.reshape(centred_vec_labels, [-1, nvec_labels, 1]) * tf.reshape(centred_vec_labels, [-1, 1, nvec_labels])
	sqsums = tf.unsorted_segment_sum(full_covs, human_labels, maxn)
	
	pack_means = means
	pack_covs = sqsums / tf.reshape(safe_weights, [maxn,1,1])
	pack_weights = weights

	predictions = batch_interaction(
		pack_means, pack_covs, pack_means, pack_covs)
	predictions = 0.99998 * predictions + 0.00001
	objective = identity_matrix(maxn)
	weight_matrix = weight_mask * tf.reshape(
		tf.sqrt(pack_weights), [-1, 1]) * tf.reshape(
		tf.sqrt(pack_weights), [1, -1])

	cost = - objective * tf.log(predictions) - \
		(1 - objective) * tf.log(1 - predictions)

	return tf.reduce_sum(weight_matrix * cost) + tf.reduce_sum(tf.reshape(central_labels_mask,[-1,1]) * bounded_cross_entropy(affinity(vec_labels, centred_vec_labels),1)), predictions

def has_error(obj, human_labels):
	obj = tf.reshape(obj, [-1])
	human_labels = tf.reshape(human_labels, [-1])
	ind = tf.to_int32(tf.argmax(obj, axis=0))
	x1=tf.equal(obj, obj[ind])
	x2=tf.equal(human_labels, human_labels[ind])
	return tf.to_float(tf.logical_not(tf.logical_or(tf.less(tf.reduce_sum(obj),1),tf.reduce_all(tf.equal(x1,x2)))))

def localized_errors(obj, human_labels, ds_shape, expander):
	return downsample([obj,human_labels], ds_shape, expander, has_error)

#f is the function applied to the downsampling window
def downsample(us, ds_shape, expander, f):
	multi=(type(us)==type([]))
	shape = ds_shape
	if multi:
		full_shape = static_shape(us[0])
	else:
		full_shape = static_shape(us)

	inds = [[i,j,k] for i in xrange(0,shape[1]) for j in xrange(0,shape[2]) for k in xrange(0,shape[3])]
	slices = [(slice(0,shape[0]),)+expander((slice(i,i+1), slice(j,j+1),slice(k,k+1))) + (slice(0,shape[4]),) for i,j,k in inds]

	if multi:
		array_form = tf.scatter_nd(indices=inds, updates=map(lambda x: f(*[V[x] for V in us]), slices), shape=shape[1:4])
	else:
		array_form = tf.scatter_nd(indices=inds, updates=map(lambda x: f(us[x]), slices), shape=shape[1:4])
	return tf.reshape(array_form,  shape )

def upsample_sum(ds, us_shape, expander):
	shape = static_shape(ds)
	full_shape = us_shape
	us = tf.Variable(tf.zeros(full_shape))
	latest = us.assign(tf.zeros(full_shape))

	inds = [[i,j,k] for i in xrange(0,shape[1]) for j in xrange(0,shape[2]) for k in xrange(0,shape[3])]
	slices = [(slice(0,shape[0]),)+expander((slice(i,i+1), slice(j,j+1),slice(k,k+1))) + (slice(0,shape[4]),) for i,j,k in inds]

	for (i,j,k),s in zip(inds, slices):
		with tf.control_dependencies([latest]):
			latest = us[s].assign(us[s]+ds[:,i,j,k,:]*tf.ones_like(us[s]))
	
	with tf.control_dependencies([latest]):
		return tf.identity(us)

def upsample_mean(ds, us_shape, expander):
	return upsample_sum(ds, us_shape, expander) / upsample_sum(tf.ones_like(ds),us_shape, expander)

def upsample_max(ds, us_shape, expander):
	shape = static_shape(ds)
	full_shape = us_shape
	us = tf.Variable(tf.zeros(full_shape))
	latest = us.assign(tf.zeros(full_shape))

	inds = [[i,j,k] for i in xrange(0,shape[1]) for j in xrange(0,shape[2]) for k in xrange(0,shape[3])]
	slices = [(slice(0,shape[0]),)+expander((slice(i,i+1), slice(j,j+1),slice(k,k+1))) + (slice(0,shape[4]),) for i,j,k in inds]

	for (i,j,k),s in zip(inds, slices):
		with tf.control_dependencies([latest]):
			latest = us[s].assign(tf.maximum(us[s],ds[:,i,j,k,:]*tf.ones_like(us[s])))
	
	with tf.control_dependencies([latest]):
		return tf.identity(us)
