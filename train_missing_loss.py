import pandas as pd
import os
from keras.preprocessing.image import img_to_array, load_img
from pathlib import Path
import math

import numpy as np

import xlrd

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from sklearn.model_selection import train_test_split

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense,Input,BatchNormalization
from keras.models import Model
from keras import optimizers, layers
from keras import backend

from mse_missing_values import missing_mse, missing_mse2



def read_and_clean_csv_files():
    global base_dir
    d2015 = pd.read_csv(os.path.join(base_dir, '2015_5_param_edit.csv'))
    d2016 = pd.read_csv(os.path.join(base_dir, '2016_5_param_edit.csv'))
    d2017 = pd.read_csv(os.path.join(base_dir, '2017_5_param_edit.csv'))
    d2018 = pd.read_csv(os.path.join(base_dir, '2018_5_param_edit.csv'))
    d2016rb  = pd.read_csv(os.path.join(base_dir, 'rb2016_5_param_edit.csv'))
    d2017rb  = pd.read_csv(os.path.join(base_dir, 'rb2017_5_param_edit.csv'))
    print("excel length:"+str(len(d2015)+len(d2016)+len(d2017)+len(d2018)+len(d2016rb)+len(d2017rb)))

    usikker_set = {'1/2', '0/1', '1/2/3', '0/1/2', '2/3', '2/3/4'}
    d2016rb.sjø = pd.Series([-1.0 if f in usikker_set else f for f in d2016rb.sjø])
    d2017rb.sjø = pd.Series([-1.0 if f in usikker_set else f for f in d2017rb.sjø])
    d2016rb.sjø = d2016rb.sjø.astype('float64')
    d2017rb.sjø = d2017rb.sjø.astype('float64')

    d2015.sjø = pd.Series( [-1.0 if (f == 0 or np.isnan(f) ) else f for f in d2015.sjø] )
    d2016.sjø = pd.Series( [-1.0 if (f == 0 or np.isnan(f) ) else f for f in d2016.sjø] )
    d2017.sjø = pd.Series( [-1.0 if (f == 0 or np.isnan(f) ) else f for f in d2017.sjø] )
    d2018.sjø = pd.Series( [-1.0 if (f == 0 or np.isnan(f) ) else f for f in d2018.sjø] )
    d2016rb.sjø = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2016rb.sjø] )
    d2017rb.sjø = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2017rb.sjø] )

    d2015.smolt = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2015.smolt] )
    d2016.smolt = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2016.smolt] )
    d2017.smolt = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2017.smolt] )
    d2018.smolt = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2018.smolt] )
    d2016rb.smolt = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2016rb.smolt] )
    d2017rb.smolt = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2017rb.smolt] )

    all = pd.DataFrame({}, columns=d2015.columns.values)
    all = pd.concat([d2015, d2016, d2017, d2018, d2016rb, d2017rb])
    assert (len(all) == len(d2015)+len(d2016)+len(d2017)+len(d2018)+len(d2016rb)+len(d2017rb))

    return d2015, d2016, d2017, d2018, d2016rb, d2017rb

def read_imr( pandas_df, img_dir, tf_images, end_count):
    global new_shape, id_column

    dir_path = os.path.join(base_dir, img_dir)
    prediktor = pd.DataFrame({}, columns=['sjø', 'smolt', 'smolt_sjø', 'gytarar', 'farmed'])

    found_count=0
    for i in range(0, len(pandas_df)):
        id = pandas_df[id_column].values[i]+'.jpg'
        path = os.path.join(dir_path, id )
        my_file = Path(path)
        if not my_file.is_file():
            path = os.path.join(dir_path, id.lower() )
            #print("path lower():"+path)
            my_file = Path(path)
        if my_file.is_file() :
            pil_img = load_img(path, grayscale=True)
            smaller_img = pil_img.resize( (new_shape[1], new_shape[0]))
            tf_images[end_count+found_count] = img_to_array(smaller_img)
            smolt = pandas_df['smolt'].values[i]
            sjo = pandas_df['sjø'].values[i]
            gytar = pandas_df['gytarar'].values[i]
            wild = pandas_df['vill/oppdrett'].values[i]
            a_pred = pd.Series({'smolt': smolt, 'sjø': sjo, 'smolt_sjø':np.array([smolt, sjo]), 'gytarar': gytar, 'farmed':wild}, name=found_count)
            prediktor = prediktor.append( a_pred )
            found_count += 1
        my_file = None

    return found_count, prediktor

new_shape = (299, 299, 1)
base_dir = '/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param'
id_column = 'ID nr.'
os.path.join(base_dir, 'hi2016_in_excel')
age = []
def do_train():
    global new_shape, age


    dataset_size_smolt_sjo = 9073
    dataset_size_smolt = 1
    dataset_size_sjo = 1
    dataset_size_gytar = 1
    dataset_size_oppdrett = 1
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    a_batch_size = 20
    add_count = 0
    rb_imgs = np.empty(shape=(9073,)+new_shape)

    d2015, d2016, d2017, d2018, d2016rb, d2017rb = read_and_clean_csv_files()

    add_count, pred15 = read_imr(d2015, 'hi2015_in_excel', rb_imgs, add_count)
    add_count, pred16 = read_imr(d2016, 'hi2016_in_excel', rb_imgs, add_count)
    add_count, pred17 = read_imr(d2017, 'hi2017_in_excel', rb_imgs, add_count)
    add_count, pred18 = read_imr(d2018, 'hi2018_in_excel', rb_imgs, add_count)
    add_count, pred16rb = read_imr(d2016rb, 'rb2016', rb_imgs, add_count)
    add_count, pred17rb = read_imr(d2017rb, 'rb2017', rb_imgs, add_count)

    two_ages15 = np.vstack( pred15.smolt_sjø.values )
    two_ages16 = np.vstack( pred16.smolt_sjø.values )
    two_ages17 = np.vstack( pred17.smolt_sjø.values )
    two_ages18 = np.vstack( pred18.smolt_sjø.values )
    two_ages16rb = np.vstack( pred16rb.smolt_sjø.values )
    two_ages17rb = np.vstack( pred17rb.smolt_sjø.values )
    age = np.concatenate( (two_ages15, two_ages16, two_ages17, two_ages18, two_ages16rb, two_ages17rb), axis=0)

    print("training set size:"+str( add_count ))

    train_set = pd.DataFrame(columns=['img', 'age'])

    train_set['img'] = pd.Series( (v[0] for v in rb_imgs) )
    for i in range(0,len(age)):
        train_set['age'].values[i] = age[i]


    print("rb_imgs:"+str(len(rb_imgs)))
    print("age:"+str(len(age)))
    train_idx, val_idx, test_idx = train_validate_test_split(range(0, len(rb_imgs)))

    print("train_idx:"+str(len(train_idx)))
    print("val_idx:"+str(len(val_idx)))
    print("test_idx:"+str(len(test_idx)))

    rb_imgs_train = np.empty(shape=(len(train_idx),)+new_shape)
    age_train = []
    for i in range(0, len(train_idx)):
        rb_imgs_train[i] = rb_imgs[train_idx[i]]
        age_train.append(age[train_idx[i]])
    age_train = np.vstack(age_train)

    rb_imgs_val = np.empty(shape=(len(val_idx),)+new_shape)
    age_val = []
    for i in range(0, len(val_idx)):
        rb_imgs_val[i] = rb_imgs[val_idx[i]]
        age_val.append(age[val_idx[i]])
    age_val=np.vstack(age_val)

    rb_imgs_test = np.empty(shape=(len(test_idx),)+new_shape)
    age_test = []
    for i in range(0, len(test_idx)):
        rb_imgs_test[i] = rb_imgs[test_idx[i]]
        age_test.append(age[test_idx[i]])
    age_test = np.vstack(age_test)

    early_stopper = EarlyStopping(patience=20)
    train_datagen = ImageDataGenerator(
        zca_whitening=True,
        width_shift_range=0.,
        height_shift_range=0., #20,
        zoom_range=0.,
        rotation_range=360,
        horizontal_flip=False,
        vertical_flip=True,
        rescale=1./255)

    train_generator = train_datagen.flow(rb_imgs_train, np.array(age_train), batch_size= a_batch_size)
    rb_imgs_val_rescaled = np.multiply(rb_imgs_val, 1./255)
    rb_imgs_test_rescaled = np.multiply(rb_imgs_test, 1./255)

    gray_model = create_inceptionV3_grayscale()
    gray_model = get_fresh_weights(gray_model)
    z = dense2_linear_output(gray_model)

    out1 = Dense(2)(z)
    out11 = Activation('linear')(z)
    out2 = Dense(2,  activation='sigmoid')(z)

    alambda = layers.Lambda(lambda x : 1+4*x)(out2)
    otolitt = Model(inputs=gray_model.input, outputs=[z])
    learning_rate=0.0004
    adam = optimizers.Adam(lr=learning_rate)
    #otolitt.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy', 'mse', 'mape'], )
    otolitt.compile(loss='mse', optimizer=adam, metrics=['accuracy', 'mse', 'mape'], )
    for layer in otolitt.layers:
        layer.trainable = True

    tensorboard = TensorBoard(log_dir='./tensorboard_missing_loss')
    checkpointer = ModelCheckpoint(
        filepath = './checkpoints_missing_loss/salmon_scale_inception.{epoch:03d}-{val_loss:.2f}.hdf5',
        verbose = 1,
        save_best_only = True,
        save_weights_only = False)

    #print("age_val:"+str(age_val))
    print("len(rb_imgs_val_rescaled):"+str(len(rb_imgs_val_rescaled)))
    print("len(age_val):"+str(len(age_val)))

    history_callback = otolitt.fit_generator(train_generator,
            steps_per_epoch=1600,
            epochs=150,
            callbacks=[early_stopper, tensorboard, checkpointer],
            validation_data=(rb_imgs_val_rescaled,  np.array(age_val)))

# use Red color channel from inceptionV3 as grayscale channel
def create_inceptionV3_grayscale():

    inception_no_sf = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3)) #Inception V3 without applying softmax
    '''Modify architecture of the InceptionV3 for grayscale data'''
    inception_no_sf_config=inception_no_sf.get_config() #Copy configuration
    gray_model_config=dict(inception_no_sf_config)
    gray_model_config['layers'][0]['config']['batch_input_shape']=(None, 299, 299, 1) #Change input shape

    inception_no_sf_weights=inception_no_sf.get_weights() #Copy weights
    gray_model_weights =inception_no_sf_weights.copy()
    gray_model_weights[0] = inception_no_sf_weights[0][:,:,0,:].reshape([3,3,1,-1]) #Only use filter for red channel for transfer learning

    gray_model=Model.from_config(gray_model_config) #Make grayscale model

    return gray_model

#It seems updating a model also updates the wegihts of the previous model. So, get fresh weights.
def get_fresh_weights(gray_model):
    gray_model.set_weights(gray_model_weights)
    return gray_model

def base_output(gray_model):
    z = gray_model.output
    z = GlobalMaxPooling2D()(z)
    z = Dense(1024)(z)
    z = Activation('relu')(z)
    return z

def dense2_linear_output(gray_model):
    z = base_output(gray_model)
    z = Dense(2, activation='linear')(z)
    return z

def dense2_sigmoid_output(gray_model):
    z = base_output(gray_model)
    z = Dense(2, activation='sigmoid')(z)
    return z

def dense1_linear_output(gray_model):
    z = base_output(gray_model)
    z = Dense(1, activation='linear')(z)
    return z

def dense1_sigmoid_output(gray_model):
    z = base_output(gray_model)
    z = Dense(1, activation='sigmoid')(z)
    return z

# Assume Salmon never stay more than 6 years in river
# (Because thats what the data tells)
def dense1_lambda_river_output(gray_model):
    z = base_output(gray_model)
    alambda = layers.Lambda(lambda x : 1+5*x)(z)
    return alambda

# Assume Salmon never live more than 12 years in sea
# (Because thats what the data tells)
def dense1_lambda_sea_output(gray_model):
    z = base_output(gray_model)
    alambda = layers.Lambda(lambda x : 1+11*x)(z)
    return alambda

# Age is integer value: Generalized linear model using Poisson
def dense1_poisson_output(gray_model):
    z = base_output(gray_model)
    z = Dense(1, activation='elu')(z)
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
    validation_and_test_split = 0.5

    df_train_x, df_notTrain_x = train_test_split(pairs, test_size = validation_and_test_set_size, random_state = a_seed)

    df_test_x, df_val_x = train_test_split(df_notTrain_x, test_size = validation_and_test_split, random_state = a_seed)

    return df_train_x, df_val_x, df_test_x

if __name__ == '__main__':
    do_train()
