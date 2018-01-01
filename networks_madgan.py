import tensorflow as tf
import numpy as np
from utils_madgan import deconv2d
from utils_madgan import conv2d
from utils_madgan import fc
from utils_madgan import leaky_relu

class MRGAN_GEN(object):
	def common_layer_G(self, z, is_train,reuse=False, z_dim=128):
		kwargs = dict()
		if 'bn' not in kwargs.keys():
			kwargs['bn'] = True
		if 'act' not in kwargs.keys():
			kwargs['act'] = tf.nn.elu

		with tf.variable_scope('generator') as scope:
			if reuse:
				scope.reuse_variables()
			h = fc('fc1', z, [z_dim, 4 * 4 * 1024], **kwargs)
			h = tf.reshape(h, [-1, 4, 4, 1024])
			h = deconv2d('deconv1', h, [-1, 8, 8, 512], [5, 5, 512, 1024], stride=2, **kwargs)
			h = deconv2d('deconv2', h, [-1, 16, 16, 256], [5, 5, 256, 512], stride=2, **kwargs)
			h = deconv2d('deconv3', h, [-1, 32, 32, 128], [5, 5, 128, 256], stride=2, **kwargs)
		return h

	def output_layer_G(self, h, isTrain, channel, name=None, reuse=False, width=64, height=64):
		with tf.variable_scope(name) as scope:
			if reuse:
				scope.reuse_variables()
			h = deconv2d('deconv4', h, [-1, height, width, 3], [5, 5, 3, 128], stride=2, act=tf.identity, bn=False)
		return h

class MRGAN_DIS(object):
	def common_layer_D(self, input_image, is_train, reuse=False):
		kwargs = dict()
		if 'bn' not in kwargs.keys():
			kwargs['bn'] = True
		if 'act' not in kwargs.keys():
			kwargs['act'] = leaky_relu

		with tf.variable_scope('discriminator') as scope:
			if reuse:
				scope.reuse_variables()
			conv1 = conv2d('conv1', input_image, [5, 5, 3, 64], stride=2, **kwargs)
			conv2 = conv2d('conv2', conv1, [5, 5, 64, 128], stride=2, **kwargs)
			conv3 = conv2d('conv3', conv2, [5, 5, 128, 256], stride=2, **kwargs)
			conv4 = conv2d('conv4', conv3, [5, 5, 256, 512], stride=2, **kwargs)

		return conv4

	def output_layer_D(self, conv4, isTrain, name=None, reuse=False):
		kwargs = dict()
		with tf.variable_scope(name) as scope:
			if reuse:
				scope.reuse_variables()
			kwargs['bn'] = False
			out = fc('fc1',
				   tf.reshape(conv4, [-1, 4 * 4 * 512]), [4 * 4 * 512, 10],
				   tf.identity, **kwargs)                                         # No BN at the last layer
		return out
