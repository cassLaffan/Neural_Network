# Same imports as the Binary file
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

# Attempted understanding of this approach brought to you by: https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/
# With its formal citation in the accompaying paper

def make_model():
	model = Sequential()
	model = add_layers.add_layers(model)
	# I have no idea why we flatten
	# Apparently the data isn't the right shape for dense layers.
	# Poor design choice
	model.add(Flatten())
	model = add_layers.make_dense_layers(model)
	# opt = optimizers.RMSprop(learning_rate=0.001, rho=0.9)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def main():
	model = make_model()
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	training = add_layers.flow_direct(datagen, 'Training_Images\\train\\', 'categorical', 64, (200, 200))
	testing = add_layers.flow_direct(datagen, 'Training_Images\\test\\', 'categorical', 64, (200, 200))
	history = model.fit_generator(training, steps_per_epoch=len(training), validation_data=testing, validation_steps=len(testing), epochs=5, verbose=9)
	_, acc = model.evaluate_generator(testing, steps=len(testing), verbose=0)
	print('> %.3f' % (acc * 100.0))
	graph_maker.summarize_diagnostics(history)

if __name__=="__main__":
	main()