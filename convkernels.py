import itertools
import tensorflow as tf
import numpy as np
from utils import constant_variable, static_shape
dtype=tf.float32
shape_dict3d={}
shape_dict2d={}
shape_dictz={}
import pprint
pp = pprint.PrettyPrinter(indent=4)

def bias_variable(schema):
	if type(schema) in [FeatureSchema]:
		return constant_variable([schema.nfeatures])
	elif type(schema) in [VectorLabelSchema]:
		return VectorLabelMap(constant_variable([schema.nlabels]),constant_variable([schema.nfeatures,schema.nlabels]))
	elif type(schema) in [LabelledFeatureSchema]:
		return FocusedMap(constant_variable([schema.nfeatures]),constant_variable([schema.nfeatures,schema.nlabels]))
	else:
		raise Exception()

class ConvKernel():
	def transpose(self):
		return TransposeKernel(self)

class ConvOuter2d():
	def __init__(self, size=(4,4), strides=(2,2), n_lower=1, n_upper=1,stddev=0.5):
		self.size=size
		self.n_lower=n_lower
		self.n_upper=n_upper
		self.strides=[1,strides[0],strides[1],1]
		initial = tf.truncated_normal([size[0],size[1],n_lower,n_upper], stddev=stddev, dtype=dtype)
		self.weights=tf.Variable(initial, dtype=dtype)
	
	def __call__(self,x,y):
		assert static_shape(x)[3] == self.n_lower
		assert static_shape(y)[3] == self.n_upper
		A=tf.nn.conv2d_backprop_filter(x, [self.size[0],self.size[1],self.n_lower,self.n_upper], y, self.strides, padding='VALID')
		return tf.reduce_sum(self.weights*A,reduction_indices=[0,1])

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

	#interaction matrix is an extra variable which modulates the strengths of the connections
	#between input and output. We assume it has size (n_lower,n_upper)
	#todo: retrieve shape information from template
	def __call__(self,x,template=None,interaction_matrix=[[1.0]]):
		with tf.name_scope('2d') as scope:
			interaction_matrix=expand_dims(interaction_matrix,[0,0])
			self.in_shape = tf.shape(x)
			tmp=tf.nn.conv2d(x,self.up_coeff*self.weights*interaction_matrix, strides=self.strides, padding='VALID')
			shape_dict2d[(tuple(tmp._shape_as_list()[0:3]), self.size, tuple(self.strides))]=tuple(x._shape_as_list()[0:3])
		return tmp

	def transpose_call(self,x,template=None,interaction_matrix=[[1.0]]):
		with tf.name_scope('2d_transpose') as scope:
			interaction_matrix=expand_dims(interaction_matrix,[0,0])
			if not hasattr(self,"in_shape"):
				self.in_shape=shape_dict2d[(tuple(x._shape_as_list()[0:3]),self.size,tuple(self.strides))]+(self.n_lower,)
			ret = tf.nn.conv2d_transpose(x, self.down_coeff*self.weights*interaction_matrix, output_shape=self.in_shape, strides=self.strides, padding='VALID')

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
	
	def __call__(self,x,template=None,interaction_matrix=[[1.0]]):
		with tf.name_scope('z') as scope:
			interaction_matrix=expand_dims(interaction_matrix,[0,0])
			xt=tf.transpose(x, perm=[1,0,2,3])
			self.in_shape=tf.shape(xt)
			tmp=tf.nn.conv2d(xt,self.up_coeff*self.weights*interaction_matrix, strides=self.strides, padding='VALID')
			shape_dictz[(tuple(tmp._shape_as_list()[0:3]),self.size)]=tuple(xt._shape_as_list()[0:3])
			ret = tf.transpose(tmp, perm=[1,0,2,3])
		return ret

	def transpose_call(self,x,template=None,interaction_matrix=[[1.0]]):
		with tf.name_scope('z_transpose') as scope:
			interaction_matrix=expand_dims(interaction_matrix,[0,0])
			xt=tf.transpose(x, perm=[1,0,2,3])
			if not hasattr(self,"in_shape"):
				self.in_shape=tuple(shape_dictz[(tuple(xt._shape_as_list()[0:3]), self.size)]+(self.n_lower,))
			tmp=tf.nn.conv2d_transpose(xt,self.down_coeff*self.weights*interaction_matrix, strides=self.strides, padding='VALID', output_shape=self.in_shape)
			ret = tf.transpose(tmp, perm=[1,0,2,3])
		return ret

class ConvKernel3d(ConvKernel):
	def __init__(self, size=(4,4,2), strides=(2,2), n_lower=1, n_upper=1, n_mid=None):
		self.size=size
		if n_mid is None:
			n_mid=n_upper
		self.kernel2d = ConvKernel2d(size=(size[0],size[1]), strides=strides, n_lower=n_lower, n_upper=n_mid)
		self.kernelz = ConvKernelZ(size=size[2], n_lower=n_mid, n_upper=n_upper)

	def __call__(self,x,template=None):
		return self.kernelz(self.kernel2d(x))

	def transpose_call(self,x,template=None):
		return self.kernel2d.transpose()(self.kernelz.transpose()(x))


class TransposeKernel(ConvKernel):
	def __init__(self,k):
		self.kernel=k
	
	def __call__(self, x, template=None):
		return self.kernel.transpose_call(x,template)

	def transpose(self):
		return self.kernel


def affinity(x, y, reduction_index=3, var=1.0):
	if x is None or y is None:
		return 1
	else:
		displacement = x - y
		interaction = tf.reduce_sum(
			displacement * displacement,
			reduction_indices=[reduction_index],
			keep_dims=True) * tf.square(var)
		return tf.exp(-0.5 * interaction)

from collections import namedtuple

class FocusedMap(namedtuple('FocusedMap', ['features','labels'])):
	def __add__(self,x):
		return FocusedMap(self.features+x.features,self.labels+x.labels)

class VectorLabelMap(namedtuple('VectorLabelMap', ['features','labels'])):
	def __add__(self,x):
		return VectorLabelMap(self.features+x.features,self.labels+x.labels)

FeatureSchema = namedtuple('FeatureSchema', ['nfeatures','level'])
LabelledFeatureSchema = namedtuple('LabelledFeatureSchema', ['nfeatures','level','nlabels'])
VectorLabelSchema = namedtuple('VectorLabelSchema', ['nfeatures','level','nlabels'])

#up is a boolean specifying whether this connection goes up or down
Connection2dSchema = namedtuple('ConnectionSchema', ['size','strides'])
Connection3dSchema = namedtuple('Connection3dSchema', ['size', 'strides'])
ConnectionTransferSchema = namedtuple('ConnectionTransferSchema',[])

class TransferConnection():
	def __init__(self,inpt_schema, otpt_schema, connection_schema):
		if inpt_schema.nfeatures == otpt_schema.nfeatures or inpt_schema.nfeatures==1:
			self.weights = constant_variable([inpt_schema.nfeatures], val=1.0)
		else:
			raise Exception()
	def __call__(self,x,template=None):
		return self.weights * x

#Returns a function which takes as input (inpt,template=None) and
#outputs otpt, where inpt and otpt have schemas inpt_schema
#and otpt_schema respectively
def connection(inpt_schema, otpt_schema, connection_schema):
	if otpt_schema.level == inpt_schema.level and connection_schema == ConnectionTransferSchema():
		return TransferConnection(inpt_schema,otpt_schema,connection_schema)

	if abs(otpt_schema.level-inpt_schema.level) != 1:
		raise Exception('inpt and otpt schema levels do not differ by 1')
	if type(connection_schema)==Connection2dSchema:
		F=ConvKernel2d
	elif type(connection_schema)==Connection3dSchema:
		F=ConvKernel3d
	if type(inpt_schema) in [FeatureSchema] and type(otpt_schema) in [FeatureSchema]:
		if otpt_schema.level == inpt_schema.level + 1:
			return F(size=connection_schema.size, strides = connection_schema.strides, n_lower = inpt_schema.nfeatures, n_upper = otpt_schema.nfeatures)
		elif otpt_schema.level == inpt_schema.level - 1:
			return F(size=connection_schema.size, strides = connection_schema.strides, n_lower = otpt_schema.nfeatures, n_upper = inpt_schema.nfeatures).transpose()
	
	if type(inpt_schema) in [LabelledFeatureSchema] and type(otpt_schema) in [LabelledFeatureSchema]:
		if otpt_schema.level == inpt_schema.level + 1:
			return FocusedConvKernel3d(size=connection_schema.size, strides = connection_schema.strides, n_lower = inpt_schema.nfeatures, n_upper=otpt_schema.nfeatures)
		else:
			return FocusedConvKernel3d(size=connection_schema.size, strides = connection_schema.strides, n_lower = otpt_schema.nfeatures, n_upper=inpt_schema.nfeatures).transpose()
	
	if type(inpt_schema) in [LabelledFeatureSchema] and type(otpt_schema) in [FeatureSchema]:
		if otpt_schema.level == inpt_schema.level + 1:
			return lambda x, template = None: F(size=connection_schema.size, strides = connection_schema.strides, n_lower = inpt_schema.nfeatures, n_upper = otpt_schema.nfeatures)(x.features,template)
		elif otpt_schema.level == inpt_schema.level - 1:
			return lambda x, template = None: F(size=connection_schema.size, strides = connection_schema.strides, n_lower = otpt_schema.nfeatures, n_upper = inpt_schema.nfeatures).transpose()(x.features,template)
	if type(inpt_schema) in [FeatureSchema] and type(otpt_schema) in [LabelledFeatureSchema]:
		if otpt_schema.level == inpt_schema.level + 1:
			return lambda x, template = None: LabelledFeatureSchema(features=F(size=connection_schema.size, strides = connection_schema.strides, n_lower = inpt_schema.nfeatures, n_upper = otpt_schema.nfeatures)(x,template),labels=None)
		elif otpt_schema.level == inpt_schema.level - 1:
			return lambda x, template = None: LabelledFeatureSchema(features=F(size=connection_schema.size, strides = connection_schema.strides, n_lower = otpt_schema.nfeatures, n_upper = inpt_schema.nfeatures).transpose()(x,template),labels=None)
	if type(inpt_schema) in [LabelledFeatureSchema] and type(otpt_schema) in [VectorLabelSchema]:
		if otpt_schema.level == inpt_schema.level + 1:
			raise Exception("not supported")
		elif otpt_schema.level == inpt_schema.level - 1:
			return VectortoFocusedKernel3d(size=connection_schema.size, strides=connection_schema.strides, n_lower = otpt_schema.nfeatures, n_upper = inpt_schema.nfeatures).transpose()
	if type(inpt_schema) in [VectorLabelSchema] and type(otpt_schema) in [LabelledFeatureSchema]:
		if otpt_schema.level == inpt_schema.level + 1:
			return VectortoFocusedKernel3d(size=connection_schema.size, strides=connection_schema.strides, n_lower = inpt_schema.nfeatures, n_upper = otpt_schema.nfeatures)
		elif otpt_schema.level == inpt_schema.level - 1:
			raise Exception("not supported")


def cast(inpt_schema,otpt_schema):
	if type(inpt_schema)==FeatureSchema and type(otpt_schema)==LabelledFeatureSchema:
		return (lambda x: FocusedMap(features=x, labels=tf.truncated_normal([inpt_schema.nfeatures, otpt_schema.nlabels])))
	elif type(inpt_schema)==FeatureSchema and type(otpt_schema)==VectorLabelSchema:
		return (lambda x: VectorLabelMap(features=x, labels=tf.truncated_normal([otpt_schema.nfeatures, otpt_schema.nlabels])))
	elif type(inpt_schema)==type(otpt_schema):
		return (lambda x:x)

class Cast():
	def __init__(self,inpt_schema, otpt_schema, stddev=0.5):
		initial = tf.truncated_normal([otpt_schema.nfeatures,otpt_schema.nlabels], stddev=stddev, dtype=dtype)
		self.weights=tf.Variable(initial, dtype=dtype)
		self.inpt_schema=inpt_schema
		self.otpt_schema=otpt_schema
	def __call__(self,x):
		inpt_schema=self.inpt_schema
		otpt_schema=self.otpt_schema
		if type(inpt_schema)==FeatureSchema and type(otpt_schema)==LabelledFeatureSchema:
			return FocusedMap(features=x, labels=self.weights)
		elif type(inpt_schema)==FeatureSchema and type(otpt_schema)==VectorLabelSchema:
			return VectorLabelMap(features=x, labels=self.weights)
		elif type(inpt_schema)==type(otpt_schema):
			return x
		else:
			raise Exception()

class MultiscaleCast():
	def __init__(self, inpt_schemas, otpt_schemas):
		self.ops = [Cast(i,o) for i,o in zip(inpt_schemas, otpt_schemas)]
	def __call__(self,x):
		return [op(a) for op,a in zip(self.ops,x)]

def expand_dims(inpt,dims):
	for count,index in enumerate(sorted(dims)):
		inpt=tf.expand_dims(inpt, count + index)
	return inpt


class FocusedConvKernel3d(ConvKernel):
	def __init__(self, size=(4,4,2), strides=(2,2), n_lower=1, n_upper=1):
		self.size=size
		n_mid=n_upper
		self.n_lower=n_lower
		self.n_upper=n_upper
		self.kernel2d = ConvKernel2d(size=(size[0],size[1]), strides=strides, n_lower=n_lower, n_upper=n_mid)
		self.kernelz = ConvKernelZ(size=size[2], n_lower=n_mid, n_upper=n_upper)
		
		self.kernelz_outer = ConvKernelZ(size=size[2], n_lower=n_mid, n_upper=n_upper)
		self.kernel2d_outer = ConvOuter2d(size=(size[0],size[1]), strides=strides, n_lower=n_lower, n_upper=n_mid)

		self.variances_lower = tf.Variable(0.1*tf.ones([n_lower,1,1],dtype=dtype))
		self.variances_upper = tf.Variable(0.1*tf.ones([1,n_upper,1],dtype=dtype))
		self.variances = self.variances_lower * self.variances_upper
		self.variances2 = self.variances_upper * tf.reshape(self.variances_upper,[n_upper,1,1])

	def __call__(self,lower, template):
		assert static_shape(lower.features)[3]==self.n_lower
		upper=template
		association_matrix_2d = affinity(expand_dims(lower.labels,[1]),expand_dims(upper.labels,[0]),reduction_index=2, var=self.variances)
		association_matrix_z = affinity(expand_dims(upper.labels,[1]),expand_dims(upper.labels,[0]),reduction_index=2, var = self.variances2)
		#labels has shape [nfeatures,nlabels]
		features = self.kernelz(self.kernel2d(lower.features,association_matrix_2d),association_matrix_z)

		label_matrix = expand_dims(self.kernel2d_outer(lower.features,self.kernelz_outer.transpose()(upper.features)),[2])
		labels = tf.reduce_sum(label_matrix*expand_dims(lower.labels,[1]),reduction_indices=[0])
		return FocusedMap(features=features,labels=labels)

	def transpose_call(self,upper, template):
		assert static_shape(upper.features)[3]==self.n_upper
		lower=template
		association_matrix_2d = affinity(expand_dims(lower.labels,[1]),expand_dims(upper.labels,[0]),reduction_index=2, var=self.variances)
		association_matrix_z = affinity(expand_dims(upper.labels,[1]),expand_dims(upper.labels,[0]),reduction_index=2, var=self.variances2)
		features = self.kernel2d.transpose()(self.kernelz.transpose()(upper.features,association_matrix_z),association_matrix_2d)

		label_matrix = expand_dims(self.kernel2d_outer(lower.features,self.kernelz_outer.transpose()(upper.features)),[2])
		labels = tf.reduce_sum(label_matrix*expand_dims(upper.labels,[0]),reduction_indices=[1])
		#todo: check dimensions
		return FocusedMap(features=features,labels=labels)

class VectortoFocused(ConvKernel):
	def __init__(self):
		pass
	def __call__(self,x):
		#x should be a  VectorLabelMap
		assert type(x)==VectorLabelMap
		labels = expand_dims(x.labels,[0,0,0])
		features = expand_dims(x.features,[3])
		affinities = tf.squeeze(affinity(features,labels, reduction_index=4),squeeze_dims=[4])

		return FocusedMap(affinities,x.labels)

	def transpose_call(self,x,template=None):
		assert type(x)==FocusedMap
		labels = expand_dims(x.labels,[0,0,0])
		features = expand_dims(x.features,[4])
		tmp = tf.reduce_sum(features*labels,reduction_indices=[3])

		return VectorLabelMap(tmp,x.labels)

class VectortoFocusedKernel3d(ConvKernel):
	def __init__(self,size=(4,4,2), strides=(2,2), n_lower=1, n_upper=1):
		self.kernel=FocusedConvKernel3d(size=size, strides=strides,n_lower=n_lower,n_upper=n_upper)
		self.op = VectortoFocused()
		self.n_lower=n_lower
		self.n_upper=n_upper
	def __call__(self,x,template):
		return self.kernel(self.op(x),template)
	def transpose_call(self,x,template):
		return self.op.transpose()(self.kernel.transpose()(x,self.op(template)))

def zero(schema):
	if type(schema)==LabelledFeatureSchema:
		return FocusedMap(0,0)
	elif type(schema)==VectorLabelSchema:
		return VectorLabelMap(0,0)
	elif type(schema) in [FeatureSchema]:
		return 0
	else:
		raise Exception()


class FocusedVectorConvKernel3d(ConvKernel):
	pass

class MultiscaleUpConv3d():
	def __init__(self, feature_schemas, connection_schemas, activations):
		n=len(feature_schemas)
		self.connections=[connection(feature_schemas[i],feature_schemas[i+1], connection_schemas[i]) for i in xrange(n-1)]
		self.biases = [bias_variable(feature_schemas[i]) for i in xrange(n)]
		self.activations=activations
	
	def __call__(self, inpt, templates=itertools.repeat(None)):
		otpts=[inpt]
		for c,template,b,a in itertools.izip(self.connections, itertools.islice(templates,1,None), itertools.islice(self.biases,1,None),itertools.islice(self.activations,1,None)):
			with tf.name_scope('up'):
				otpts.append(a(c(otpts[-1],template=template)+b))
		return otpts

class MultiscaleDownConv3d():
	def __init__(self, feature_schemas, connection_schemas, activations):
		n=len(feature_schemas)
		self.connections=[connection(feature_schemas[i+1],feature_schemas[i], connection_schemas[i]) for i in xrange(n-1)]
		self.biases = [bias_variable(feature_schemas[i]) for i in xrange(n)]
		self.activations=activations
	
	def __call__(self, inpt, templates=itertools.repeat(None)):
		otpts=[inpt]
		for c,template,b,a in itertools.izip(self.connections, itertools.islice(templates,1,None), itertools.islice(self.biases,1,None),itertools.islice(self.activations,1,None)):
			with tf.name_scope('up'):
				otpts.insert(0,a(c(otpts[0],template=template)+b))
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
						(z if i==n-1 else self.diag_connections[i](inpt[i+1], template = inpt[i])),
						(z if i==0 else self.up_connections[i-1](ret[i-1], template=inpt[i])),
						self.transfer_connections[i](inpt[i], template=inpt[i])
						]
				ret[i] = sum(l,z)
				ret[i] = self.activations[i](ret[i] + self.biases[i])
		return ret
