#Korsbrekk2016-161.jpg JPEG 2048x1536 2048x1536+0+0 8-bit DirectClass 906KB 0.000u 0:00.000
#131.jpg JPEG 2560x1920 2560x1920+0+0 8-bit DirectClass 741KB 0.000u 0:00.000

#/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param/rundlesing2020

#/gpfs/gpfs0/deep/projects/em-salmon-scales/checkpoints_best_salmon_sea_batch_16


import numpy as np
import pandas as pd
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import scipy
import tensorflow as tf
from keras.models import load_model

from clean_y_true import read_and_clean_4_param_csv
from train_util import read_images, load_xy, get_checkpoint_tensorboard, create_model_grayscale, get_fresh_weights, base_output, dense1_linear_output, train_validate_test_split


from efficientnet import EfficientNetB4

new_shape = (380, 380, 3)
IMG_SHAPE = (380, 380)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

dir_path = "/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param/rundlesing2020"
dir_path = "/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param/ringlesning2019"
max_dataset_size =  len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])

ringlesing_imgs = np.empty(shape=(10,)+new_shape)
ringlesing_imgs, filename = read_images_from_xvalidationset(dir_path, ringlesing_imgs)
sea_age_model = load_model('/gpfs/gpfs0/deep/projects/em-salmon-scales/checkpoints_best_salmon_sea_batch_16/salmon_scale_efficientnetB4.046-0.29.hdf5')

ringlesing_imgs = np.multiply(ringlesing_imgs, 1./255)
y_hat = sea_age_model.predict(ringlesing_imgs)

for x, y in zip(filename, y_hat):
    print (x, y[0])


# test on test_set
do_test_sea()

def read_images_from_xvalidationset(dir_path, rb_imgs):
    
    found_count=0
    filename=list()
    for image_name in os.listdir(dir_path):
        if image_name in {'9.jpg','10.jpg','11.jpg','12.jpg','13.jpg','14.jpg','15.jpg','16.jpg','17.jpg','18.jpg'}:
            path = os.path.join(dir_path, image_name )
            pil_img = load_img(path, target_size=IMG_SHAPE, grayscale=False)
            array_img = img_to_array(pil_img, data_format='channels_last')
            rb_imgs[found_count] = array_img
            filename.append(image_name)
            found_count += 1
    
    return rb_imgs, filename
    
def do_test_sea():
    rb_imgs, all_sea_age, all_smolt_age, all_farmed_class, all_spawn_class, all_filenames = load_xy()    
    
uten_ukjent = len(all_sea_age) - all_sea_age.count(-1.0)
rb_imgs2 = np.empty(shape=(uten_ukjent,)+new_shape)
unique, counts = np.unique(all_sea_age, return_counts=True)
print("age distrib:"+str( dict(zip(unique, counts)) ))

all_sea_age2 = []
found_count = 0
all_filenames2 = []
for i in range(0, len(all_sea_age)):
    if all_sea_age[i] > -1:        
        rb_imgs2[found_count] = rb_imgs[i]
        all_sea_age2.append(all_sea_age[i])
        found_count += 1
        all_filenames2.append(all_filenames[i])

assert found_count == uten_ukjent

age = all_sea_age2
rb_imgs = rb_imgs2

train_idx, val_idx, test_idx = train_validate_test_split( range(0, len(rb_imgs)) )

test_rb_imgs = np.empty(shape=(len(test_idx),)+new_shape)
test_age = []
test_age_names = []
for i in range(0, len(test_idx)):
    test_rb_imgs[i] = rb_imgs[test_idx[i]]
    test_age.append(age[test_idx[i]])
    test_age_names.append(all_filenames2[test_idx[i]])
    
test_rb_imgs = np.multiply(test_rb_imgs, 1./255)    
y_hat = sea_age_model.predict(test_rb_imgs)