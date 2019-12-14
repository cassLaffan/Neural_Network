from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
import pandas as pd
import json
import numpy as np
from random import seed
from random import random
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# I should really comment these things because now I'm wondering what the heck this code does
# Seriously wtf?
# training_data = pd.read_csv("Training_Data.csv")

# Sequential programming? I hardly know her.
# Upon programming this, it has become way tidier than my binary sorting one
# I typed that before I smashed the functions together

photos, labels = [], []

seed(1)
val_ratio = 0.25
src_directory = 'Training_Images\\'

with open('Training_Data.json') as json_file:
    data = json.load(json_file)
    for p in data:
        output = 0
        if p['Label']['objects'][0]['title']=='Cat':
            output = int(p['Label']['objects'][0]['classifications'][0]['answer']['value'])

        file = p['Label']['objects'][0]['featureId'] + ".jpg"

        photo = load_img(src_directory + file)
        photo = img_to_array(photo)
        photos.append(photo)

        labels.append(output)

        src = src_directory + '\\' + file
        dst_dir = 'train\\'
        if random() < val_ratio:
            dst_dir = 'test\\'
        if(output >= 1 and output <= 5):
            dst = src_directory + dst_dir + 'cat_' + str(output) + file
            copyfile(src, dst)
        elif output==0:
            dst = src_directory + dst_dir + 'not_cats\\'  + file
            copyfile(src, dst)


# convert to a numpy arrays
labels = np.asarray(labels)
# save the reshaped photos
np.save('cats_photos.npy', photos)
np.save('cats_labels.npy', labels)



