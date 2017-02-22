import tensorflow as tf
from utils import *

def bounded_cross_entropy(guess,truth):
	guess = 0.999998*guess + 0.000001
	return  - truth * tf.log(guess) - (1-truth) * tf.log(1-guess)

def label_diff(x,y):
	return tf.to_float(tf.equal(x,y))


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

