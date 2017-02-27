from convkernels3d import *
from activations import *
from utils import *

def patch_size_suggestions(top_shape):
	f0=range_tuple_expander(strides=(1,2,2),sizes=(1,4,4))
	f1=range_tuple_expander(strides=(1,2,2),sizes=(4,4,4))
	f2=range_tuple_expander(strides=(2,2,2),sizes=(4,4,4))
	f3=range_tuple_expander(strides=(2,2,2),sizes=(4,4,4))
	f4=range_tuple_expander(strides=(2,2,2),sizes=(4,4,4))

	base_patch_size = shape_to_slices(top_shape)
	patch_size_suggestions = list(reversed(map(slices_to_shape,cum_compose(f4,f3,f2,f1,f0)(base_patch_size))))
	print(patch_size_suggestions)
	return patch_size_suggestions

def make_forward_net(patch_size, n_in, n_out):

	feature_schemas = [
				FeatureSchema(n_in+n_out,1),
				FeatureSchema(24,2),
				FeatureSchema(28,3),
				FeatureSchema(32,4),
				FeatureSchema(48,5),
				FeatureSchema(64,6),
				]
	connection_schemas = [
				Connection2dSchema(size=(4,4),strides=(2,2)),
				Connection3dFactorizedSchema(size=(4,4,4),strides=(1,2,2)),
				Connection3dFactorizedSchema(size=(4,4,4),strides=(2,2,2)),
				Connection3dFactorizedSchema(size=(4,4,4),strides=(2,2,2)),
				Connection3dFactorizedSchema(size=(4,4,4),strides=(2,2,2))
				]

	initial_activations = [
		lambda x: x,
		tf.nn.elu,
		tf.nn.elu,
		tf.nn.elu,
		tf.nn.elu,
		tf.nn.elu,
		tf.nn.elu
		]

	activations = [
		tf.nn.sigmoid,
		tf.nn.elu,
		tf.nn.elu,
		tf.nn.elu,
		tf.nn.elu,
		tf.nn.elu,
		tf.nn.elu
		]

	initial = MultiscaleUpConv3d(
			feature_schemas = feature_schemas,
			connection_schemas = connection_schemas,
		activations=initial_activations)
	it1 = MultiscaleConv3d(feature_schemas, feature_schemas, connection_schemas, connection_schemas, activations)
	it2 = MultiscaleConv3d(feature_schemas, feature_schemas, connection_schemas, connection_schemas, activations)
	it3 = MultiscaleConv3d(feature_schemas, feature_schemas, connection_schemas, connection_schemas, activations)
	it4 = MultiscaleConv3d(feature_schemas, feature_schemas, connection_schemas, connection_schemas, activations)
	it5 = MultiscaleConv3d(feature_schemas, feature_schemas, connection_schemas, connection_schemas, activations)
	it6 = MultiscaleConv3d(feature_schemas, feature_schemas, connection_schemas, connection_schemas, activations)


	ds_it1_pre = MultiscaleConv3d(feature_schemas[1:], feature_schemas[1:], connection_schemas[1:], connection_schemas[1:], activations[1:])
	ds_it2_pre = MultiscaleConv3d(feature_schemas[1:], feature_schemas[1:], connection_schemas[1:], connection_schemas[1:], activations[1:])
	
	ds_it1 = lambda l: l[0:1] + ds_it1_pre(l[1:])
	ds_it2 = lambda l: l[0:1] + ds_it2_pre(l[1:])

	fl1 = FullLinear(n_in=64, n_out=1)
	lin1 = lambda x: tf.reshape(fl1(x),[])
	lin2 = FullLinear(n_in=48, n_out=1)

	def discriminate(x):
		with tf.name_scope("discriminate"):
			padded_x = tf.concat([x,tf.zeros((1,) + patch_size + (n_out,))],4)
			tower = compose(
					initial,
					it1,
					it2,
					ds_it1,
					ds_it2,
					)(padded_x)
			discrim = lin1(reduce_spatial(tower[-1]))
			discrim_mid = lin2(tower[-2])
		return discrim, discrim_mid


	def reconstruct(x):
		with tf.name_scope("forward"):
			padded_x = tf.concat([x,tf.zeros((1,) + patch_size + (n_out,))],4)
			i=initial(padded_x)
			print(map(static_shape,i))
			return compose(
					it1,
					it2,

					ds_it1,
					ds_it2,
					ds_it1,
					ds_it2,

					it3,
					it4,

					)(i)[0][:,:,:,:,n_in:n_in+n_out]
	return discriminate, reconstruct
