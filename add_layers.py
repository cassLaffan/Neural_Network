from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
import numpy
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10

# This function was meant to remove redundancy but I think I just moved the redundancy into its own function
# Takes the model and returns the model
def add_layers(model):
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(BatchNormalization())
	model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(BatchNormalization())
	model.add(Conv2D(8, (3, 3), activation='relu',  padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(BatchNormalization())
	model.add(Conv2D(4, (3, 3), activation='relu',  padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.5))
	return model

# Called so I can make as many dense layers as I want (you can't tell me what to do)
# Takes the model and returns the model
def make_dense_layers(model):
	model.add(Dense(100, activation='relu'))
	model.add(Dense(75, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(25, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(6, activation='softmax'))
	return model

# gen has to be of type Image Data generator, target needs to be a tuple of the pixel size of the images
def flow_direct(datagen, path, mode, batch, target):
	direct = datagen.flow_from_directory(path, class_mode=mode, batch_size=batch, target_size=target)
	return direct