import tensorflow.contrib.slim as slim
import tensorflow as tf

def build_model(input, d=20, c=64, f=3, r=9):
	network = input
	for i in range(d-1):
		Bi = tf.Variable(tf.zeros([c]))
		network = tf.nn.relu(slim.conv2d(network, c, [f,f], activation_fn=None) + Bi)
	Bd = tf.Variable(tf.zeros([3*r]))
	network = slim.conv2d(network, 3*r, [f,f], activation_fn=None) + Bd
	subchannels = tf.split(network,r,3)
	for s in range(r):
		subchannels[s] = subchannels[s] + input
	network = tf.concat(subchannels, 3)
	return network
