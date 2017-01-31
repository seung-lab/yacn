import tensorflow as tf
import numpy as np

def conv2d(full,template):
	return tf.nn.conv2d(full,template,[1,1,1,1],padding="VALID")

def normxcorr(full,template):
	full = full - tf.reduce_mean(full)
	eps=0.001

	#whiten template
	template = template - tf.reduce_mean(template)
	template = template / (tf.sqrt(tf.reduce_mean(tf.square(template)))+eps)

	ones=tf.ones_like(template)
	N=tf.reduce_sum(ones)
	sqfull=tf.square(full)

	means = conv2d(full, ones)/N
	sqmeans = conv2d(sqfull, ones)/N
	stds = tf.sqrt(sqmeans - tf.square(means))

	return conv2d(full,template)/(N*stds+eps)

sess=tf.Session()

with tf.device("/gpu:0"):
	full=tf.placeholder(tf.float32,shape=[1,512,512,1])
	template=tf.placeholder(tf.float32,shape=[25,25,1,1])
	result=normxcorr(full,template)
import time

sess.run(result,feed_dict={
	full: np.ones([1,512,512,1]).astype(np.float32),
	template: np.ones([25,25,1,1]).astype(np.float32),
	})
x1=time.time()
n=10
for i in xrange(n):
	sess.run(result,feed_dict={
		full: np.ones([1,512,512,1]).astype(np.float32),
		template: np.ones([25,25,1,1]).astype(np.float32),
		})
x2=time.time()
print((x2-x1)/n)
