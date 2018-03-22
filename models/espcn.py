import tensorflow.contrib.slim as slim
import tensorflow as tf

def build_model(input, f1=5, f2=3, n1=64, n2=32):
	B1 = tf.Variable(tf.zeros([n1]))
	B2 = tf.Variable(tf.zeros([n2]))
	B3 = tf.Variable(tf.zeros([1]))
	network = tf.nn.relu(slim.conv2d(input, n1, [f1,f1], activation_fn=None) + B1)
	network = tf.nn.relu(slim.conv2d(network, n2, [1,1], activation_fn=None) + B2)
	network = slim.conv2d(network, 3, [f2,f2], activation_fn=None) + B3
	return network
