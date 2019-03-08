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
#import keras.layers.Lambda

from sklearn.model_selection import train_test_split

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense,Input,BatchNormalization
from keras.models import Model
from keras import optimizers, layers
from keras import backend

from v3_grayscale_missing_loss import baseline_model4

def missing_mse(y_true, y_pred):
    print("********** IN MISSING_MSE***********")
    # y_pred=tf.constant([[1.0,2.0],[5.0,10.0]])

    float_missing = K.cast(tf.logical_not(tf.equal(y_pred,-1.0)), dtype='float32')
    y_true = K.print_tensor(y_true, message='y_true = ')
    y_pred = K.print_tensor(y_pred, message='y_pred = ')
    #bool_missing = Lambda((lambda x: tf.Print(bool_missing, [bool_missing], message='message', first_n=-1, summarize=1024)), name='name')(layer)

    return K.mean(K.square( (y_pred - y_true)*float_missing ), axis=-1)


age = []
new_shape = (299, 299, 1)
def do_train():
    global new_shape, age

    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    base_dir = '/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param'
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

    null_alder = {'0'}
    d2015.sjø = pd.Series( [-1.0 if (f in null_alder or np.isnan(f)) else f for f in d2015.sjø] )
    d2016.sjø = pd.Series( [-1.0 if (f in null_alder or np.isnan(f)) else f for f in d2016.sjø] )
    d2017.sjø = pd.Series( [-1.0 if (f in null_alder or np.isnan(f)) else f for f in d2017.sjø] )
    d2018.sjø = pd.Series( [-1.0 if (f in null_alder or np.isnan(f)) else f for f in d2018.sjø] )
    print(d2018.sjø.dtype)
    print(d2016rb.sjø.dtype)
    d2016rb.sjø = pd.Series( [-1.0 if (f in null_alder or np.isnan(f)) else f for f in d2016rb.sjø] )
    d2017rb.sjø = pd.Series( [-1.0 if (f in null_alder or np.isnan(f)) else f for f in d2017rb.sjø] )

    d2015.smolt = pd.Series( [-1.0 if (f in null_alder or np.isnan(f)) else f for f in d2015.smolt] )
    d2016.smolt = pd.Series( [-1.0 if (f in null_alder or np.isnan(f)) else f for f in d2016.smolt] )
    d2017.smolt = pd.Series( [-1.0 if (f in null_alder or np.isnan(f)) else f for f in d2017.smolt] )
    d2018.smolt = pd.Series( [-1.0 if (f in null_alder or np.isnan(f)) else f for f in d2018.smolt] )
    d2016rb.smolt = pd.Series( [-1.0 if (f in null_alder or np.isnan(f)) else f for f in d2016rb.smolt] )
    d2017rb.smolt = pd.Series( [-1.0 if (f in null_alder or np.isnan(f)) else f for f in d2017rb.smolt] )

    column = 'sjø'
    count_rb = 0
    rb_imgs = np.empty(shape=(400,)+new_shape) #699

    add_count = 0
    count15 = read_any_img(d2015, os.path.join(base_dir, 'hi2015_in_excel'), 'ID nr.', rb_imgs, count_rb, age)

    add_count = count15
    count16 = read_imr(d2016, os.path.join(base_dir, 'hi2016_in_excel'), 'ID nr.', rb_imgs, add_count, age, column)

    add_count = count15+count16
    count17 = read_imr(d2017, os.path.join(base_dir, 'hi2017_in_excel'), 'ID nr.', rb_imgs, add_count, age, column)

    add_count = count15+count16+count17
    count18 = read_imr(d2018, os.path.join(base_dir, 'hi2018_in_excel'), 'ID nr.', rb_imgs, add_count, age, column)

    add_count = count15+count16+count17+count18
    count16rb = read_imr(d2016rb, os.path.join(base_dir, 'rb2016'), 'ID nr.', rb_imgs, add_count, age, column)

    add_count= count15+count16+count17+count18+count16rb
    count17rb = read_imr(d2017rb, os.path.join(base_dir, 'rb2017'), 'ID nr.', rb_imgs, add_count, age, column)

    num_ex = count16rb + count17rb + count15 + count16 + count17 + count18
    print("training set size:"+str( num_ex ))
    print("len age:"+str(len(age)))
    print("2015:"+str(count15))
    print("2016:"+str(count16))
    print("2017:"+str(count17))
    print("2018:"+str(count18))
    print("rb2016:"+str(count16rb))
    print("rb2017:"+str(count17rb))

    a_batch_size = 20
    train_set=None
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
    #print(age)

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

    inception = baseline_model4()
    z = inception.output
    #out1 = Dense(1,  activation='linear')(z)
    out2 = Dense(2,  activation='sigmoid')(z)
    alambda = layers.Lambda(lambda x : 1+4*x)(out2)
    otolitt = Model(inputs=inception.input, outputs=[alambda])
    learning_rate=0.0004
    adam = optimizers.Adam(lr=learning_rate)
    #otolitt.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy', 'mse', 'mape'], )
    otolitt.compile(loss=missing_mse, optimizer=adam, metrics=['accuracy', 'mse', 'mape'], )
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
            steps_per_epoch=1600,#4000
            epochs=150,
            #callbacks=[early_stopper, tensorboard, checkpointer],
            callbacks=[early_stopper, tensorboard, checkpointer],
            validation_data=(rb_imgs_val_rescaled,  np.array(age_val)))
            #validation_data=val_generator,
            #validation_steps=len(val_generator))

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

def read_imr(pandas_df, dir_path, id_column, np_images, end_count, list_age, param_to_read):
    global new_shape, age

    found_count=0
    pandas_df[param_to_read] = pandas_df[param_to_read].dropna()
    for i in range(0, len(pandas_df)):
        if not pd.isnull(pandas_df[param_to_read].values[i]):
            if pandas_df[param_to_read].values[i] != None and not pd.isnull(pandas_df[id_column].values[i]):
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
                    np_images[end_count+found_count] = img_to_array(smaller_img)
                    age.append([pandas_df['smolt'].values[i], pandas_df['sjø'].values[i]])
                    found_count += 1
                my_file = None
    return found_count

def read_any_img(pandas_df, dir_path, id_column, np_images, end_count, list_age):
    global new_shape, age

    found_count=0
    smolt_age = []
    sjo_age = []
    #new_age=np.array([[]])
    for i in range(0, len(pandas_df)):
        if len(age) < 400:
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
                np_images[end_count+found_count] = img_to_array(smaller_img)
                age.append(np.array([pandas_df['smolt'].values[i], pandas_df['sjø'].values[i]]))
                #smolt_age.append(pandas_df['smolt'].values[i])
                #sjo_age.append(pandas_df['sjø'].values[i])
                found_count += 1
                my_file = None

    age = np.vstack(age)
    #age.append([smolt_age, sjo_age])
    return found_count




if __name__ == '__main__':
    do_train()
