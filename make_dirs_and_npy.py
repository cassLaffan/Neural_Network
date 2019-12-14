import pandas as pd
import json
import numpy
from random import seed
from random import random
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

training_data = pd.read_csv("Training_Data.csv")
json_list = []

for index, row in training_data.iterrows():
    json_list.append(json.loads(row["Label"]))

photos, labels = [], []

seed(1)
val_ratio = 0.25
src_directory = 'Training_Images\\'

for item in json_list:
    output = 0.0
    if item['objects'][0]['value']=="cat":
        output = 1.0
    file = item['objects'][0]["featureId"] + ".jpg"
    photo = load_img(src_directory + file)
    photo = img_to_array(photo)
    photos.append(photo)
    labels.append(output)

    src = src_directory + '\\' + file
    dst_dir = 'train\\'
    if random() < val_ratio:
        dst_dir = 'test\\'
    if output==1.0:
        dst = src_directory + dst_dir + 'cats\\'  + file
        copyfile(src, dst)
    elif output==0.0:
        dst = src_directory + dst_dir + 'not_cats\\'  + file
        copyfile(src, dst)

# convert to a numpy arrays
labels = asarray(labels)
# save the reshaped photos
save('cats_photos.npy', photos)
save('cats_labels.npy', labels)