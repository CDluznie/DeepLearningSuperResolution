import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import time
import os
import utils
import models.espcn
import models.edspcn

input_image_size = 50
output_image_size = 100
scaling_factor = output_image_size // input_image_size

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def PeriodicShuffle(network, f=3):
	batchsize = tf.shape(network)[0]
	periodic_shuffle_channels = 3*(scaling_factor*scaling_factor)
	network = slim.conv2d(network, periodic_shuffle_channels, [f,f], activation_fn=None)
	channels = tf.split(network, 3, 3)
	for c in range(3):
		channels[c] = tf.reshape(channels[c], [batchsize, input_image_size, input_image_size, scaling_factor, scaling_factor])
		channels[c] = tf.transpose(channels[c], (0, 1, 3, 2, 4))
		channels[c] = tf.reshape(channels[c], [batchsize, output_image_size, output_image_size, 1])
	network = tf.concat(channels, 3)
	return network

class SuperResolutionNeuralNetwork(object):

	def create(modelType):
		if modelType == "espcn":
			model_generator = models.espcn.build_model
		elif modelType == "edspcn":
			model_generator = models.edspcn.build_model
		else:
			raise ValueError("invalid model type " + modelType)
		return SuperResolutionNeuralNetwork(os.path.join("models", "save", modelType), model_generator)

	def __init__(self, save_path, model_generator):
		self.path = save_path
		self.input = tf.placeholder(tf.float32,[None,input_image_size,input_image_size,3])
		self.target = tf.placeholder(tf.float32,[None,output_image_size,output_image_size,3])
		image_input = self.input - tf.reduce_mean(self.input)
		image_target = self.target - tf.reduce_mean(self.input)
		image_output = PeriodicShuffle(model_generator(image_input))
		self.output = tf.clip_by_value(image_output + tf.reduce_mean(self.input), 0.0, 255.0)
		self.loss = tf.reduce_mean(tf.squared_difference(image_target, image_output))		
		self.PSNR = utils.PSNR(image_target, image_output)
		self.optimizer =  tf.train.AdamOptimizer()	
		self.session = tf.Session()
		self.saver = tf.train.Saver()

	def save(self):
		self.saver.save(self.session, os.path.join(self.path, "model"))

	def restore(self):
		self.saver.restore(self.session, tf.train.latest_checkpoint(self.path))

	def upscale(self, x):
		return self.session.run(self.output, feed_dict={self.input:[x]})[0]
	
	def training_summary(self, maxImage=5):
		tf.summary.scalar("loss", self.loss)
		tf.summary.scalar("PSNR", self.PSNR)
		tf.summary.image("input", tf.cast(self.input,tf.uint8), max_outputs=maxImage)
		tf.summary.image("target", tf.cast(self.target,tf.uint8), max_outputs=maxImage)
		tf.summary.image("output", tf.cast(self.output,tf.uint8), max_outputs=maxImage)

	def train(self, dataset, batchsize, iterations):	
		summary_path = os.path.join(self.path, "train")
		for summary_file in os.listdir(summary_path):
			os.remove(os.path.join(summary_path, summary_file))
		self.training_summary()
		summary_op = tf.summary.merge_all()
		train_op = self.optimizer.minimize(self.loss)	
		init = tf.global_variables_initializer()
		self.session.run(init)
		train_writer = tf.summary.FileWriter(summary_path, self.session.graph)
		total_time = 0
		for iteration in range(iterations):
			batch = dataset.pick(batchsize)
			start_time = time.time()
			_,summary,loss_val,PSNR_val = self.session.run(
				[train_op, summary_op, self.loss, self.PSNR],
				{
					self.input:[learningData.input for learningData in batch],
					self.target:[learningData.target for learningData in batch]
				}
			)
			end_time = time.time()
			current_time = end_time - start_time
			print(
				("| Epoch: %d | Loss: %.4f | PSNR: %.4f | Time: %.4f |") % 
				(iteration+1, loss_val, PSNR_val, current_time)
			)
			train_writer.add_summary(summary,iteration)
			total_time += current_time
		print("Total training time : %.4f seconds" % total_time)
