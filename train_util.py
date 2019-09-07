import numpy as np
import pandas as pd
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import scipy

import tensorflow as tf

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.optimizers import SGD
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Activation, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers, layers
from keras import backend as K

from clean_y_true import read_and_clean_4_param_csv

base_dir = '/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param'
id_column = 'ID nr.'
max_dataset_size = 9073
SEA_AGE = 'sjø'
SMOLT_AGE = 'smolt'
FARMED_CLASS = 'vill'
SPAWN_CLASS = 'gytarar'
new_shape = (380, 380, 3)
IMG_SHAPE = (380, 380)

def read_images(pandas_df, rb_imgs, array_pointer, directory_with_images):
    global base_dir, id_column, SEA_AGE, SMOLT_AGE, FARMED_CLASS, SPAWN_CLASS

    filename=list()
    sea_age = list()
    smolt_age = list()
    farmed_class = list()
    spawn_class = list()
    found_count=0
    dir_path = os.path.join(base_dir, directory_with_images)
    print("path:"+dir_path+ " first file:"+str(pandas_df[id_column].values[0]))
    for i in range(0, len(pandas_df)):
        image_name = pandas_df[id_column].values[i]+'.jpg'
        path = os.path.join(dir_path, image_name )
        my_file = Path(path)
        if not my_file.is_file():
            path = os.path.join(dir_path, image_name.lower() )
            my_file = Path(path)
        if my_file.is_file():
            pil_img = load_img(path, target_size=IMG_SHAPE, grayscale=False)
            array_img = img_to_array(pil_img, data_format='channels_last')
            rb_imgs[array_pointer+found_count] = array_img
            sea_age.append( pandas_df[SEA_AGE].values[i] )
            smolt_age.append( pandas_df[SMOLT_AGE].values[i] )
            farmed_class.append( pandas_df[FARMED_CLASS].values[i] )
            spawn_class.append( pandas_df[SPAWN_CLASS].values[i] )
            filename.append(my_file)
            found_count += 1

    end_of_array = array_pointer + found_count
    return end_of_array, rb_imgs, sea_age, smolt_age, farmed_class, spawn_class, filename

def load_xy():
    global new_shape, max_dataset_size, base_dir

    rb_imgs = np.empty(shape=(max_dataset_size,)+new_shape)
    d2015, d2016, d2017, d2018, d2016rb, d2017rb = read_and_clean_4_param_csv( base_dir )

    cumulativ_add_count = 0

    cumulativ_add_count, rb_imgs, d15_sea_age, d15_smolt_age, d15_farmed_class, d15_spawn_class, filenames1 = \
        read_images(d2015, rb_imgs, cumulativ_add_count, 'hi2015_in_excel')
    print("cumulativ_add_count_15: "+str(cumulativ_add_count))
    cumulativ_add_count, rb_imgs, d16_sea_age, d16_smolt_age, d16_farmed_class, d16_spawn_class, filenames2 = \
        read_images(d2016, rb_imgs, cumulativ_add_count, 'hi2016_in_excel')
    print("cumulativ_add_count_16: "+str(cumulativ_add_count))
    cumulativ_add_count, rb_imgs, d17_sea_age, d17_smolt_age, d17_farmed_class, d17_spawn_class, filenames3 = \
        read_images(d2017, rb_imgs, cumulativ_add_count, 'hi2017_in_excel')
    print("cumulativ_add_count_17 :"+str(cumulativ_add_count))
    cumulativ_add_count, rb_imgs, d18_sea_age, d18_smolt_age, d18_farmed_class, d18_spawn_class, filenames4 = \
        read_images(d2018, rb_imgs, cumulativ_add_count, 'hi2018_in_excel')
    print("cumulativ_add_count_18 :"+str(cumulativ_add_count))
    cumulativ_add_count, rb_imgs, d16rb_sea_age, d16rb_smolt_age, d16rb_farmed_class, d16rb_spawn_class, filenames5 = \
        read_images(d2016rb, rb_imgs, cumulativ_add_count, 'rb2016')
    print("cumulativ_add_count_16rb: "+str(cumulativ_add_count))
    cumulativ_add_count, rb_imgs, d17rb_sea_age, d17rb_smolt_age, d17rb_farmed_class, d17rb_spawn_class, filenames6 = \
        read_images(d2017rb, rb_imgs, cumulativ_add_count, 'rb2017')
    print("cumulativ_add_count_17rb: "+str(cumulativ_add_count))

    all_sea_age = d15_sea_age + d16_sea_age + d17_sea_age + d18_sea_age + d16rb_sea_age + d17rb_sea_age
    all_smolt_age = d15_smolt_age + d16_smolt_age + d17_smolt_age + d18_smolt_age + d16rb_smolt_age + d17rb_smolt_age
    all_farmed_class = d15_farmed_class + d16_farmed_class + d17_farmed_class + d18_farmed_class + d16rb_farmed_class + d17rb_farmed_class
    all_spawn_class = d15_spawn_class + d16_spawn_class + d17_spawn_class + d18_spawn_class + d16rb_spawn_class + d17rb_spawn_class
    all_filenames = filenames1 + filenames2+filenames3+filenames4+filenames5+filenames6

    return rb_imgs, all_sea_age, all_smolt_age, all_farmed_class, all_spawn_class, all_filenames

def get_checkpoint_tensorboard(tensorboard_path, checkpoint_path):

    tensorboard = TensorBoard(log_dir=tensorboard_path)
    checkpointer = ModelCheckpoint(
        filepath = checkpoint_path,
        verbose = 1,
        save_best_only = True,
        save_weights_only = False)
    return tensorboard, checkpointer

def create_model_grayscale(new_shape):

    model_no_sf = EfficientNetB4(include_top=False, weights='imagenet', input_shape=new_shape, classes=2)

    #inception_no_sf = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3)) #Inception V3 without applying softmax
    '''Modify architecture of the InceptionV3 for grayscale data'''
    model_no_sf_config=model_no_sf.get_config() #Copy configuration
    gray_model_config=dict(model_no_sf_config)
    gray_model_config['layers'][0]['config']['batch_input_shape']=((None,)+ new_shape) #Change input shape

    model_no_sf_weights=model_no_sf.get_weights() #Copy weights
    gray_model_weights =model_no_sf_weights.copy()
    gray_model_weights[0] = model_no_sf_weights[0][:,:,0,:].reshape([3,3,1,-1]) #Only use filter for red channel for transfer learning

    gray_model=Model.from_config(gray_model_config) #Make grayscale model

    return gray_model, gray_model_weights

def get_fresh_weights(model, model_weights):
    model.set_weights(model_weights)
    return model

def base_output(model):
    z = model.output
    z = GlobalMaxPooling2D()(z)
    return z

def dense_classification_softmax(model):
    z = base_output(model)
    z = Dense(2)(z)
    z = Activation('softmax')(z)
    return z

def dense_classification_sigmoid(model):
    z = base_output(model)
    z = Dense(2)(z)
    z = Activation('sigmoid')(z)
    return z

def dense1_linear_output(gray_model):
    z = base_output(gray_model)
    z = Dense(1, activation='linear')(z)
    return z

def train_validate_test_split(pairs, validation_set_size = 0.15, test_set_size = 0.15, a_seed = 8):
    """ split pairs into 3 set, train-, validation-, and test-set
        1 - (validation_set_size + test_set_size) = % training set size
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = np.array([np.arange(10)]*2).T  # 2 columns for x, y, and one for index
    >>> df_ = pd.DataFrame(data, columns=['x', 'y'])
    >>> train_x, val_x, test_x = \
             train_validate_test_split( df_, validation_set_size = 0.2, test_set_size = 0.2, a_seed = 1 )
    >>> train_x['x'].values
    array([0, 3, 1, 7, 8, 5])
    >>> val_x['x'].values
    array([4, 6])
    >>> test_x['x'].values
    array([2, 9])
    """
    validation_and_test_set_size = validation_set_size + test_set_size
    validation_and_test_split = validation_set_size / (test_set_size+validation_set_size)

    df_train_x, df_notTrain_x = train_test_split(pairs, test_size = validation_and_test_set_size, random_state = a_seed)

    df_test_x, df_val_x = train_test_split(df_notTrain_x, test_size = validation_and_test_split, random_state = a_seed)

    return df_train_x, df_val_x, df_test_x
