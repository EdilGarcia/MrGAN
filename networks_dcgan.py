import tensorflow as tf
import numpy as np

class MRGAN_GEN(object):
	def lrelu(self, x, n, leak=0.2):
		return tf.maximum(x, leak * x, name=n)

	def common_layer_G(self, z, is_train=True,reuse=False, z_dim=128):
		c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32
		s4 = 4
		with tf.variable_scope('generator') as scope:
			if reuse:
				scope.reuse_variables()
			w1 = tf.get_variable('w1', shape=[z_dim, s4 * s4 * c4], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02))
			b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,initializer=tf.constant_initializer(0.0))
			flat_conv1 = tf.add(tf.matmul(z, w1), b1, name='flat_conv1')

			conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
			bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
			# act1 = tf.nn.relu(bn1, name='act1')
			act1 = self.lrelu(bn1, n='act1')
			# 8*8*256
			conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
											   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
											   name='conv2')
			bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
			# act2 = tf.nn.relu(bn2, name='act2')
			act2 =  self.lrelu(bn2, n='act2')
			# 16*16*128
			conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
											   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
											   name='conv3')
			bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
			# act3 = tf.nn.relu(bn3, name='act3')
			act3 =  self.lrelu(bn3, n='act3')
			# 32*32*64
			conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
											   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
											   name='conv4')
			bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
			# act4 = tf.nn.relu(bn4, name='act4')
			act4 =  self.lrelu(bn4, n='act4')
			# 64*64*32
			conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
											   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
											   name='conv5')
			bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
			# act5 = tf.nn.relu(bn5, name='act5')
			act5 =  self.lrelu(bn5, n='act5')
		return act5

	def output_layer_G(self, act5, isTrain, channel, name=None, reuse=False):
		with tf.variable_scope(name) as scope:
			if reuse:
				scope.reuse_variables()
			conv6 = tf.layers.conv2d_transpose(act5, 3, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
											   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
											   name='conv6')
			act6 = tf.nn.tanh(conv6, name='act6')
		return act6

class MRGAN_DIS(object):
	def lrelu(self, x, n, leak=0.2):
		return tf.maximum(x, leak * x, name=n)

	def common_layer_D(self, input_image, is_train, reuse=False):
		c2, c4, c8, c16 = 64, 128, 256, 512
		with tf.variable_scope('discriminator') as scope:
			if reuse:
				scope.reuse_variables()

			conv1 = tf.layers.conv2d(input_image, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
									 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
									 name='conv1')
			bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
			#act1 =  tf.maximum(conv1, 0.2 * conv1)
			act1 = self.lrelu(conv1, n='act1')
			 #Convolution, activation, bias, repeat!
			conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
									 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
									 name='conv2')
			bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
			act2 = self.lrelu(bn2, n='act2')
			#Convolution, activation, bias, repeat!
			conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
									 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
									 name='conv3')
			bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
			act3 = self.lrelu(bn3, n='act3')
			 #Convolution, activation, bias, repeat!
			conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
									 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
									 name='conv4')
			bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
			act4 = self.lrelu(bn4, n='act4')
		return	act4

	def output_layer_D(self, act4, isTrain, name=None, reuse=False):
		with tf.variable_scope(name) as scope:
			if reuse:
				scope.reuse_variables()
			dim = int(np.prod(act4.get_shape()[1:]))
			fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')
			w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
								initializer=tf.truncated_normal_initializer(stddev=0.02))
			b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
								initializer=tf.constant_initializer(0.0))
			# wgan just get rid of the sigmoid
			logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
		return logits
