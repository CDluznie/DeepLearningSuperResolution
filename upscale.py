import scipy.misc
import argparse
import network as nw
from network import SuperResolutionNeuralNetwork
import utils
import numpy as np
import os
import sys

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Upscale a colored image with neural network")
	parser.add_argument("--model",	default="edspcn")
	parser.add_argument("--image", 	required=True)
	args = parser.parse_args()

	input_image_name, input_image_extension = os.path.splitext(os.path.basename(args.image))
	input_image = scipy.misc.imread(args.image)
	crop_width = nw.output_image_size * (input_image.shape[1] // nw.output_image_size)
	crop_height = nw.output_image_size * (input_image.shape[0] // nw.output_image_size)
	input_image = utils.crop(input_image, crop_width, crop_height)
	image_height,image_width,_ = input_image.shape
	
	print("Building " + args.model.upper() + " network...")
	network = SuperResolutionNeuralNetwork.create(args.model)
	print("Networkd builded")

	print("Restoring network...")
	network.restore()
	print("Network restored")

	lr_output_image = scipy.misc.imresize(input_image,(nw.scaling_factor*image_height,nw.scaling_factor*image_width), interp="bicubic")
	hr_output_image = np.empty([nw.scaling_factor*image_height, nw.scaling_factor*image_width, 3])

	print("Upscaling image...")
	tills_height = image_height // nw.input_image_size
	tills_width = image_width // nw.input_image_size
	for i in range(tills_height):
		for j in range(tills_width):
			till = input_image[
				i*nw.input_image_size : (i+1)*nw.input_image_size,
				j*nw.input_image_size : (j+1)*nw.input_image_size
			]
			till_hr = network.upscale(till)
			hr_output_image[
				i*nw.output_image_size : (i+1)*nw.output_image_size,
				j*nw.output_image_size : (j+1)*nw.output_image_size
			] = till_hr
	print("Image upscaled")

	scipy.misc.imsave(os.path.join("results", input_image_name + "_lr" + input_image_extension), lr_output_image)
	scipy.misc.imsave(os.path.join("results", input_image_name + "_hr" + input_image_extension), hr_output_image)
