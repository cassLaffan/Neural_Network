from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense

# This function was meant to remove redundancy but I think I just moved the redundancy into its own function
# Takes the model and returns the model
def add_layers(model):
	# Don't use this 128 one of you value your CPU not being melted onto your motherboard
	# model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	# model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	# model.add(MaxPooling2D((2, 2)))
	return model

# Called so I can make as many dense layers as I want (you can't tell me what to do)
# Takes the model and returns the model
def make_dense_layers(model):
    model.add(Dense(160, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# gen has to be of type Image Data generator, target needs to be a tuple of the pixel size of the images
def flow_direct(datagen, path, mode, batch, target):
	direct = datagen.flow_from_directory(path, class_mode=mode, batch_size=batch, target_size=target)
	return direct