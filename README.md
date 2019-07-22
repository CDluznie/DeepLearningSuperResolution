# Deep Learning Super-Resolution

The goal of this project is to create a deep neural network for single image super-resolution.
To solve this problem, we implement a state-of-the-art network (efficient sub-pixel convolutional neural networkESPCN) and build our own from that one (efficient deep sub-pixel convolutional neural network EDSPCN).

## Results

#### Input image :
![image-bicubic](./examples/car_BICUBIC.bmp)

#### ESPCN output image :
![image-bicubic](./examples/car_ESPCN.bmp)

#### EDSPCN output image :
![image-bicubic](./examples/car_EDSPCN.bmp)

## Requirements

* Python 3
* TensorFlow
* NumPy
* SciPy

## Usage

* **Training** : `python3 train.py`
  * `--model model` : model to train, possible values : espcn, edspcn (edspcn by default)
  * `--dataset dataset` : training dataset (*data/General-100* by default)
  * `--batchsize batchsize` : size of batch (20 by default)
  * `--epochs epochs` : number of epochs (1000 by default)

* **Super-Resolution** : `upscale.py`
  * `--model model` : model to use, possible values : espcn, edspcn (edspcn by default)
  * `--image image` : input image to upscale
    
* **Summary of the training stage** : `tensorboard --logdir models/save/*MODEL*/train`
