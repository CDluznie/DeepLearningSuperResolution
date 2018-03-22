import dataset
import argparse
import network
from network import SuperResolutionNeuralNetwork

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Train a super-resolution neural network")
	parser.add_argument("--model",		default="edspcn")
	parser.add_argument("--dataset",	default="data/General-100")
	parser.add_argument("--batchsize",	default=20,	type=int)
	parser.add_argument("--epochs",	default=1000,	type=int)
	args = parser.parse_args()

	print("Loading dataset...")
	ds = dataset.DataSet(args.dataset, network.input_image_size, network.output_image_size)
	print("Dataset loaded")
	print("Dataset contains " + str(len(ds)) + " examples")

	print("Building " + args.model.upper() + " network...")
	network = SuperResolutionNeuralNetwork.create(args.model)
	print("Networkd builded")

	print("Training network for " + str(args.epochs) + " epochs...")
	network.train(ds, args.batchsize, args.epochs)
	print("Network trained")

	print("Saving network...")
	network.save()
	print("Network saved")
