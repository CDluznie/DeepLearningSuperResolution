import tensorflow as tf

def tills(img, size, noverlap) :
	lst = list()
	h,w,_ = img.shape
	shift = size//noverlap
	for y in range (0, h-size+1, shift):
		for x in range(0, w-size+1, shift):
			till = img[y:y+size, x:x+size]
			lst.append(till)
	return lst

def crop(img, width, height):
	h,w,_ = img.shape
	x = w//2 - width//2
	y = h//2 - height//2    
	return img[y:y+height, x:x+width]

def log10(x):
  return tf.log(x)/tf.log(tf.constant(10, dtype=tf.float32))

def PSNR(img1, img2):
	return tf.constant(10,dtype=tf.float32)*log10(tf.constant(255**2,dtype=tf.float32)/tf.reduce_mean(tf.squared_difference(img1,img2)))
