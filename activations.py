import tensorflow as tf
dtype=tf.float32

class SymmetricTanh():
	def __init__(self,reduction_index=3):
		self.premultiply = tf.Variable(1.0, dtype=dtype)
		self.postmultiply = tf.Variable(1.0, dtype=dtype)
		self.reduction_index=reduction_index
		
	def __call__(self,x):
		with tf.name_scope("symmetric_tanh"):
			x=self.premultiply*x
			lengths = tf.sqrt(tf.reduce_sum(tf.square(x),reduction_indices=[self.reduction_index],keep_dims=True))
			lengths = tf.maximum(lengths,0.0001)
			return self.postmultiply*x*tf.tanh(lengths)/lengths
