import scipy.misc
import matplotlib.pyplot as plt
import os
import random
import utils

class LearningData:
	def __init__(self, target, input):
		self.input = input
		self.target = target

class DataSet:
	
	def __init__(self, relpath, input_image_size, output_image_size, overlap=3):
		self.data = list()
		images = os.listdir(relpath);
		for image in images:
			img = scipy.misc.imread(os.path.join(relpath, image))
			h,w,_ = img.shape
			if h < output_image_size or w < output_image_size:
				continue
			crop_width = output_image_size * (w // output_image_size)
			crop_height = output_image_size * (h // output_image_size)
			img = utils.crop(img, crop_width, crop_height)
			for till in utils.tills(img, output_image_size, overlap):
				lr_till = scipy.misc.imresize(till,(input_image_size,input_image_size))
				self.add(LearningData(till,lr_till))
	
	def add(self, learningdata):
		self.data.append(learningdata)

	def __len__(self):
		return len(self.data)

	def show_data(self):
		for img in self.data:
			plt.imshow(img.target)
			plt.show()
			plt.imshow(img.input)
			plt.show()
			
	def pick(self, n):
		picked_data = list()
		for i in range(n):
			picked_data.append(random.choice(self.data))
		return picked_data

