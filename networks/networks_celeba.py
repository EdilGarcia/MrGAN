import tensorflow as tf
import numpy as np
from utils import lrelu

class MRGAN_GEN(object):
	def shared_G(self, z, is_train=True,reuse=False, z_dim=128):
		with tf.variable_scope('generator') as scope:
			if reuse:
				scope.reuse_variables()
			w1 = tf.get_variable('w1', shape=[z_dim, 2 * 2 * 1024], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02))
			b1 = tf.get_variable('b1', shape=[1024 * 2 * 2], dtype=tf.float32,initializer=tf.constant_initializer(0.0))
			flat_conv1 = tf.add(tf.matmul(z, w1), b1, name='flat_conv1')

			conv1 = tf.reshape(flat_conv1, shape=[-1, 2, 2, 1024], name='conv1')
			bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
			act1 = lrelu(bn1, n='act1')


			conv2 = tf.layers.conv2d_transpose(act1, 512, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
											   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
											   name='conv2')
			bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
			act2 =  lrelu(bn2, n='act2')

			conv3 = tf.layers.conv2d_transpose(act2, 256, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
											   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
											   name='conv3')
			bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
			act3 =  lrelu(bn3, n='act3')

			conv4 = tf.layers.conv2d_transpose(act3, 128, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
											   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
											   name='conv4')
			bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
			act4 =  lrelu(bn4, n='act4')
		return act4

	def unshared_G(self, act4, is_train, channel, name=None, reuse=False):
		with tf.variable_scope(name) as scope:
			if reuse:
				scope.reuse_variables()

			conv5 = tf.layers.conv2d_transpose(act4, 64, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
											   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
											   name='conv5')
			bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
			act5 =  lrelu(bn5, n='act5')

			conv6 = tf.layers.conv2d_transpose(act5, 32, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
											   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
											   name='conv6')
			bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
			act6 =  lrelu(bn6, n='act6')

			conv7 = tf.layers.conv2d_transpose(act6, 3, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
											   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
											   name='conv7')
			act7 = tf.nn.tanh(conv7, name='act6')
		return act7

class MRGAN_DIS(object):
	def unshared_D(self, input_image, is_train, name=None, reuse=False):
		with tf.variable_scope(name) as scope:
			if reuse:
				scope.reuse_variables()

			conv1 = tf.layers.conv2d(input_image, 64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
									 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
									 name='conv1')
			bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
			act1 = lrelu(conv1, n='act1')
			 #Convolution, activation, bias, repeat!
			conv2 = tf.layers.conv2d(act1, 128, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
									 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
									 name='conv2')
			bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
			act2 = lrelu(bn2, n='act2')
			#Convolution, activation, bias, repeat!
			conv3 = tf.layers.conv2d(act2, 256, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
									 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
									 name='conv3')
			bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
			act3 = lrelu(bn3, n='act3')
			 #Convolution, activation, bias, repeat!
			conv4 = tf.layers.conv2d(act3, 512, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
									 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
									 name='conv4')
			bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
			act4 = lrelu(bn4, n='act4')
		return	act4

	def shared_D(self, act4, is_train, name='discriminator', reuse=False):
		with tf.variable_scope(name) as scope:
			if reuse:
				scope.reuse_variables()
			conv5 = tf.layers.conv2d(act4, 1024, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
									 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
									 name='conv5')
			bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
			act5 = lrelu(bn5, n='act5')

			dim = int(np.prod(act5.get_shape()[1:]))
			fc1 = tf.reshape(act5, shape=[-1, dim], name='fc1')
			w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
								initializer=tf.truncated_normal_initializer(stddev=0.02))
			b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
								initializer=tf.constant_initializer(0.0))
			# wgan just get rid of the sigmoid
			logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
		return logits
