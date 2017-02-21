import tensorflow as tf
import numpy as np
dtype=tf.float32
shape_dict3d={}
shape_dict2d={}
shape_dictz={}
import operator
from collections import namedtuple
import itertools


def make_variable(shape, val=0.0):
	initial = tf.constant(val, dtype=dtype, shape=shape)
	var = tf.Variable(initial, dtype=dtype)
	return var

def bias_variable(schema):
	if type(schema) in [FeatureSchema]:
		return make_variable([schema.nfeatures])
	else:
		raise Exception()

class ConvKernel():
	def transpose(self):
		return TransposeKernel(self)

class TransposeKernel(ConvKernel):
	def __init__(self,k):
		self.kernel=k
	
	def __call__(self, x):
		return self.kernel.transpose_call(x)

	def transpose(self):
		return self.kernel

class ConvKernel2d(ConvKernel):
	def __init__(self, size=(4,4), strides=(2,2), n_lower=1, n_upper=1,stddev=0.5):
		initial = tf.truncated_normal([size[0],size[1],n_lower,n_upper], stddev=stddev, dtype=dtype)
		self.weights=tf.Variable(initial, dtype=dtype)
		self.size=size
		self.strides=[1,strides[0],strides[1],1]
		self.n_lower=n_lower
		self.n_upper=n_upper
		self.up_coeff = 1.0/np.sqrt(size[0]*size[1]*n_lower)
		self.down_coeff = 1.0/np.sqrt((size[0]*size[1])/(strides[0]*strides[1])*n_upper)
	
	def transpose(self):
		return TransposeKernel(self)

	def __call__(self,x):
		with tf.name_scope('2d') as scope:
			x=tf.squeeze(x,0)
			tmp=tf.nn.conv2d(x,self.up_coeff*self.weights, strides=self.strides, padding='VALID')
			shape_dict2d[(tuple(tmp._shape_as_list()[0:3]), self.size, tuple(self.strides))]=tuple(x._shape_as_list()[0:3])
			tmp=tf.expand_dims(x,0)
		return tmp

	def transpose_call(self,x):
		with tf.name_scope('2d_transpose') as scope:
			if not hasattr(self,"in_shape"):
				self.in_shape=shape_dict2d[(tuple(x._shape_as_list()[0:3]),self.size,tuple(self.strides))]+(self.n_lower,)
			ret = tf.nn.conv2d_transpose(x, self.down_coeff*self.weights, output_shape=self.in_shape, strides=self.strides, padding='VALID')

		return ret

class ConvKernelZ(ConvKernel):
	def __init__(self, size=2, n_lower=1, n_upper=1, stddev=0.5):
		initial = tf.truncated_normal([size,1,n_lower,n_upper], stddev=stddev, dtype=dtype)
		self.weights=tf.Variable(initial, dtype=dtype)
		self.size=size
		self.strides=[1,1,1,1]
		self.n_lower=n_lower
		self.n_upper=n_upper
		self.up_coeff = 1.0/np.sqrt(size*n_lower)
		self.down_coeff = 1.0/np.sqrt(size*n_upper)
	
	def __call__(self,x):
		with tf.name_scope('z') as scope:
			x=tf.squeeze(x,0)
			xt=tf.transpose(x, perm=[1,0,2,3])
			tmp=tf.nn.conv2d(xt,self.up_coeff*self.weights, strides=self.strides, padding='VALID')
			shape_dictz[(tuple(tmp._shape_as_list()[0:3]),self.size)]=tuple(xt._shape_as_list()[0:3])
			ret = tf.transpose(tmp, perm=[1,0,2,3])
			ret=tf.expand_dims(x,0)
		return ret

	def transpose_call(self,x):
		with tf.name_scope('z_transpose') as scope:
			x=tf.squeeze(x,0)
			xt=tf.transpose(x, perm=[1,0,2,3])
			if not hasattr(self,"in_shape"):
				self.in_shape=tuple(shape_dictz[(tuple(xt._shape_as_list()[0:3]), self.size)]+(self.n_lower,))
			tmp=tf.nn.conv2d_transpose(xt,self.down_coeff*self.weights, strides=self.strides, padding='VALID', output_shape=self.in_shape)
			ret = tf.transpose(tmp, perm=[1,0,2,3])
			ret=tf.expand_dims(x,0)
		return ret

class ConvKernel3dFactorized_Old(ConvKernel):
	def __init__(self, size=(2,4,4), strides=(1,2,2), n_lower=1, n_upper=1, n_mid=None):
		self.size=size
		if n_mid is None:
			n_mid=n_upper
		self.kernel2d = ConvKernel2d(size=(size[1],size[2]), strides=strides[1:3], n_lower=n_lower, n_upper=n_mid)
		self.kernelz = ConvKernelZ(size=size[1], n_lower=n_mid, n_upper=n_upper)

	def __call__(self,x):
		return self.kernelz(self.kernel2d(x))

	def transpose_call(self,x):
		return self.kernel2d.transpose()(self.kernelz.transpose()(x))

class ConvKernel3d(ConvKernel):
	def __init__(self, size=(1,4,4), strides=(1,2,2), n_lower=1, n_upper=1,stddev=0.5,dtype=dtype):
		initial = tf.truncated_normal([size[0],size[1],size[2],n_lower,n_upper], stddev=stddev, dtype=dtype)
		self.weights=tf.Variable(initial, dtype=dtype)
		self.size=size
		self.strides=[1,strides[0],strides[1],strides[2],1]
		self.n_lower=n_lower
		self.n_upper=n_upper

		#up_coeff and down_coeff are coefficients meant to keep the magnitude of the output independent of stride and size choices
		self.up_coeff = 1.0/np.sqrt(reduce(operator.mul,size)*n_lower)
		self.down_coeff = 1.0/np.sqrt(reduce(operator.mul,size)/(reduce(operator.mul,strides))*n_upper)
	
	def transpose(self):
		return TransposeKernel(self)

	def __call__(self,x):
		with tf.name_scope('conv3d') as scope:
			tmp=tf.nn.conv3d(x, self.up_coeff*self.weights, strides=self.strides, padding='VALID')
			shape_dict3d[(tuple(tmp._shape_as_list()[1:4]), self.size, tuple(self.strides))]=tuple(x._shape_as_list()[1:4])
		return tmp

	def transpose_call(self,x):
		with tf.name_scope('conv3d_t') as scope:
			if not hasattr(self,"in_shape"):
				self.in_shape=shape_dict3d[(tuple(x._shape_as_list()[1:4]),self.size,tuple(self.strides))]+(self.n_lower,)
			full_in_shape = (x._shape_as_list()[0],)+self.in_shape
			ret = tf.nn.conv3d_transpose(x, self.down_coeff*self.weights, output_shape=full_in_shape, strides=self.strides, padding='VALID')

		return ret


class ConvKernel3dFactorized(ConvKernel):
	def __init__(self, size=(1,4,4), strides=(1,2,2), n_lower=1, n_mid=1, n_upper=1, stddev=0.5, dtype=dtype):
		self.kernelxy = ConvKernel3d(size=(1,size[1],size[2]), strides=(1,strides[1],strides[2]), n_lower=n_lower, n_upper=n_mid, stddev=stddev, dtype=dtype)
		self.kernelz = ConvKernel3d(size=(size[0],1,1), strides=(strides[0],1,1), n_lower=n_mid, n_upper=n_upper, stddev=stddev, dtype=dtype)
	
	def transpose(self):
		return TransposeKernel(self)

	def __call__(self,x):
		with tf.name_scope('conv3d_factorized') as scope:
			return self.kernelz(self.kernelxy(x))

	def transpose_call(self,x):
		with tf.name_scope('conv3d_factorized_t') as scope:
			return self.kernelxy.transpose()(self.kernelz.transpose()(x))

class TransferConnection():
	def __init__(self,inpt_schema, otpt_schema, connection_schema):
		if inpt_schema.nfeatures == otpt_schema.nfeatures or inpt_schema.nfeatures==1:
			self.weights = make_variable([inpt_schema.nfeatures], val=1.0)
		else:
			raise Exception()
	def __call__(self,x):
		return self.weights * x

FeatureSchema = namedtuple('FeatureSchema', ['nfeatures','level'])
Connection3dSchema = namedtuple('Connection3dSchema', ['size', 'strides'])
Connection3dFactorizedSchema = namedtuple('Connection3dFactorizedSchema', ['size', 'strides'])
ConnectionTransferSchema = namedtuple('ConnectionTransferSchema',[])

def connection(inpt_schema, otpt_schema, connection_schema):
	if otpt_schema.level == inpt_schema.level and connection_schema == ConnectionTransferSchema():
		return TransferConnection(inpt_schema,otpt_schema,connection_schema)
	if type(connection_schema) in [Connection3dSchema, Connection3dFactorizedSchema]:
		if type(connection_schema) is Connection3dSchema:
			F=ConvKernel3d
		elif type(connection_schema) is Connection3dFactorizedSchema:
			F=ConvKernel3dFactorized
		if type(inpt_schema) in [FeatureSchema] and type(otpt_schema) in [FeatureSchema]:
			if otpt_schema.level == inpt_schema.level + 1:
				return F(size=connection_schema.size, strides = connection_schema.strides, n_lower = inpt_schema.nfeatures, n_upper = otpt_schema.nfeatures)
			elif otpt_schema.level == inpt_schema.level - 1:
				return F(size=connection_schema.size, strides = connection_schema.strides, n_lower = otpt_schema.nfeatures, n_upper = inpt_schema.nfeatures).transpose()

def zero(schema):
	if type(schema) in [FeatureSchema]:
		return 0
	else:
		raise Exception()

class MultiscaleUpConv3d():
	def __init__(self, feature_schemas, connection_schemas, activations):
		n=len(feature_schemas)
		self.connections=[connection(feature_schemas[i],feature_schemas[i+1], connection_schemas[i]) for i in xrange(n-1)]
		self.biases = [bias_variable(feature_schemas[i]) for i in xrange(n)]
		self.activations=activations
	
	def __call__(self, inpt):
		otpts=[inpt]
		for c,b,a in itertools.izip(self.connections, itertools.islice(self.biases,1,None),itertools.islice(self.activations,1,None)):
			with tf.name_scope('up'):
				otpts.append(a(c(otpts[-1])+b))
		return otpts


class MultiscaleDownConv3d():
	def __init__(self, feature_schemas, connection_schemas, activations):
		n=len(feature_schemas)
		self.connections=[connection(feature_schemas[i+1],feature_schemas[i], connection_schemas[i]) for i in xrange(n-1)]
		self.biases = [bias_variable(feature_schemas[i]) for i in xrange(n)]
		self.activations=activations
	
	def __call__(self, inpt):
		otpts=[inpt]
		for c,b,a in itertools.izip(self.connections, itertools.islice(self.biases,1,None),itertools.islice(self.activations,1,None)):
			with tf.name_scope('up'):
				otpts.insert(0,a(c(otpts[0])+b))
		return otpts

class MultiscaleConv3d():
	def __init__(self, inpt_schemas, otpt_schemas, diagonal_schemas, up_schemas, activations, transfer_schemas = itertools.repeat(ConnectionTransferSchema())):
		n=len(inpt_schemas)
		self.n=n
		assert len(otpt_schemas) == n
		transfer_schemas = [x for x in itertools.islice(transfer_schemas,n)]
		self.diag_connections = [connection(inpt_schemas[i+1],otpt_schemas[i],diagonal_schemas[i]) for i in xrange(n-1)]
		self.up_connections = [connection(otpt_schemas[i],otpt_schemas[i+1],up_schemas[i]) for i in xrange(n-1)]
		self.transfer_connections = [connection(inpt_schemas[i],otpt_schemas[i],transfer_schemas[i]) for i in xrange(n)]
		self.activations = activations
		self.biases = [bias_variable(otpt_schemas[i]) for i in xrange(n)]
		self.otpt_schemas = otpt_schemas
		self.inpt_schemas = inpt_schemas

	def __call__(self, inpt):
		n=self.n
		ret = [0 for i in xrange(n)]
		for i in xrange(n):
			with tf.name_scope('unit'+str(i)):
				z=zero(self.otpt_schemas[i])
				l = [
						(z if i==n-1 else self.diag_connections[i](inpt[i+1])),
						(z if i==0 else self.up_connections[i-1](ret[i-1])),
						self.transfer_connections[i](inpt[i])
						]
				ret[i] = sum(l,z)
				ret[i] = self.activations[i](ret[i] + self.biases[i])
		return ret

"""
tmp=tf.zeros([1,100,100,100,5])
c=ConvKernel3dFactorized(size=(18,4,4), strides=(1,2,2), n_lower=5, n_mid=2, n_upper=7, stddev=0.5, dtype=dtype)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tmp).shape)
print(sess.run(c(tmp)).shape)
print(sess.run(c.transpose()(c(tmp))).shape)
"""