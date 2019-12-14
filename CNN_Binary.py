# Import Regular things
import numpy as np
import requests
from io import BytesIO
from shutil import copyfile
# Import fancy ML things
from numpy import asarray
from numpy import save
import keras.backend as K
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Flatten
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
# Import things from other files because refactoring makes me feel productive
import graph_maker
import add_layers

# def custom_loss(yTrue,yPred):
# 	#implementation of a hinge-loss esque... function
# 	return K.sum(K.max(0, (yTrue - yPred + 1)))

# What's on the label
# (Kind of) Understanding dense layers thanks to: https://medium.com/@hunterheidenreich/understanding-keras-dense-layers-2abadff9b990
# Proper citation is in the citations section of the accompanying paper
def make_model():
	model = Sequential()
	model = add_layers.add_layers(model)
	# I have no idea why we flatten
	# Every example online has it though and when I took it out, it stopped working
	model.add(Flatten())
	model = add_layers.make_dense_layers(model)
	# opt = SGD(lr=0.001, momentum=0.9)
	opt = optimizers.RMSprop(learning_rate=0.001, rho=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

## Main agorithm courtesy of: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
## And the Keras documentation
def main():
	model = make_model()
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	training = add_layers.flow_direct(datagen, 'Training_Images\\train\\', 'binary', 64, (200, 200))
	testing = add_layers.flow_direct(datagen, 'Training_Images\\test\\', 'binary', 64, (200, 200))
	history = model.fit_generator(training, steps_per_epoch=len(training), validation_data=testing, validation_steps=len(testing), epochs=20, verbose=0)
	_, acc = model.evaluate_generator(testing, steps=len(testing), verbose=0)
	print('> %.3f' % (acc * 100.0))
	graph_maker.summarize_diagnostics(history)

if __name__=="__main__":
	main()