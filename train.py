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

from sklearn.model_selection import train_test_split

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense,Input,BatchNormalization
from keras.models import Model
from keras import optimizers, layers

from v3_grayscale import baseline_model4

df  = pd.read_csv('Totalark_2016_Malde.tsv', sep='\t', skiprows=4)

def get_full_filepath():
    count = 0
    rb_path_total = pd.DataFrame(columns=['ID nr.', 'Totalt'])
    for i in range(0, len(df)):
        file_name = df['ID nr.'].values[i]
        total = df['Totalt'].values[i]
        fylke = df['Fylke'].values[i]
        from_RB_root = '/gpfs/gpfs0/deep/data/salmon-scales/from_RB/'+fylke
        found = False
        if not pd.isnull(total):
            for root, dirs, files in os.walk(from_RB_root):
                if found == True:
                    break
                for name in files:
                    if name == file_name+'.jpg':
                        #print(file_name)
                        #print(root + '/'+name + " - "+str(total))
                        #print(fylke)
                        path_to_jpg = root + '/'+name
                        dict = {'ID nr.': path_to_jpg, 'Totalt': total}
                        rb_path_total = rb_path_total.append(dict, ignore_index=True)
                        count += 1
                        found = True

new_shape = (299, 299, 1)
def do_train():
    d2014 = pd.read_csv('data_2014.csv')
    d2015 = pd.read_csv('data_2015.csv')
    d2016 = pd.read_csv('data_2016.csv')
    #d2017 = pd.read_csv('data_2017.csv')
    d2017 = pd.read_csv('data_2017_new2.csv')
    d2016rb  = pd.read_csv('rb_path_total.csv')
    d2017rb  = pd.read_csv('rb_path_total_2017.csv')

    d2014 = d2014.dropna()
    d2015 = d2015.dropna()
    d2016 = d2016.dropna()

    count_rb = 0
    age = []
    rb_imgs = np.empty(shape=(7829,)+new_shape)
    for i in range(0, len(d2016rb) ):
        path = d2016rb['ID nr.'].values[i]
        count_rb +=1
        pil_img = load_img(path, grayscale=True)
        smaller_img = pil_img.resize( (new_shape[1], new_shape[0]))
        rb_imgs[i] = img_to_array(smaller_img)
        age.append(d2016rb['Totalt'].values[i])
        
    for i in range(0, len(d2017rb) ):
        path = d2017rb['ID nr.'].values[i]
        count_rb +=1
        pil_img = load_img(path, grayscale=True)
        smaller_img = pil_img.resize( (new_shape[1], new_shape[0]))
        rb_imgs[i] = img_to_array(smaller_img)
        age.append(d2017rb['Totalt'].values[i])    

    count17=0
    count17 = read_imr(
        d2017, 
        '/data/delphi/alle/Laks 2014-2015-2016-2017/X 2017 X/X BILDER X 2017', 
        'ID nr.', 
        rb_imgs,
        count_rb,
        age)

    count16=0
    count16 = read_imr(
        d2016, 
        '/data/delphi/alle/Laks 2014-2015-2016-2017/X 2016 X/X BILDER 2016 X', 
        'ID nr.', 
        rb_imgs,
        count_rb,
        age)

    count15=0
    count15 = read_imr(
        d2015, 
        '/data/delphi/alle/Laks 2014-2015-2016-2017/X 2015 X/X BILDER 2015 X', 
        'Idnummer', 
        rb_imgs,
        count_rb+count16,
        age)
        
    count14=0
    count14 = read_imr(
        d2014, 
        '/data/delphi/alle/Laks 2014-2015-2016-2017/X 2014 X/X BILDER 2014 X', 
        'ID nr.', 
        rb_imgs,
        count_rb+count16+count15,
        age)

    num_ex = count_rb + count14 + count15 + count16 + count17
    print("training set size:", num_ex)

    a_batch_size = 20
    train_set=None
    train_set = pd.DataFrame(columns=['img', 'age'])

    train_set['img'] = pd.Series( (v[0] for v in rb_imgs) )
    for i in range(0,len(age)):
        train_set['age'].values[i] = age[i]

    train_idx, val_idx, test_idx = train_validate_test_split(range(0, len(rb_imgs)))

    rb_imgs_train = np.empty(shape=(len(train_idx),)+new_shape)
    age_train = []
    for i in range(0, len(train_idx)):
        rb_imgs_train[i] = rb_imgs[train_idx[i]]
        age_train.append(age[train_idx[i]])

    rb_imgs_val = np.empty(shape=(len(val_idx),)+new_shape)
    age_val = []
    for i in range(0, len(val_idx)):
        rb_imgs_val[i] = rb_imgs[val_idx[i]]
        age_val.append(age[val_idx[i]])

    rb_imgs_test = np.empty(shape=(len(test_idx),)+new_shape)
    age_test = []
    for i in range(0, len(test_idx)):
        rb_imgs_test[i] = rb_imgs[test_idx[i]]
        age_test.append(age[test_idx[i]])

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
    otolitt = Model(inputs=inception.input, outputs=z)
    learning_rate=0.0004
    adam = optimizers.Adam(lr=learning_rate)
    otolitt.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy', 'mse', 'mape'])
    for layer in otolitt.layers:
        layer.trainable = True

    #inception = InceptionV3(include_top=False, weights='imagenet', input_shape=new_shape)
        
    #z = inception.output
    #z = GlobalAveragePooling2D()(z)
    #z = Dense(1024)(z)
    #z = Dropout(0.5)(z)
    #z = Activation('relu')(z)
    #z = Dense(1, input_dim=1024)(z)
    #z = Dense(1)(z)
    #z = Activation('linear')(z)
    #otolitt = Model(inputs=inception.input, outputs=z)

    #adam = optimizers.Adam(lr=learning_rate, decay= decay)
    #otolitt.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy', 'mse', 'mape'])

    #add weight decay
    #for layer in my_model.layers:
    #    if hasattr(layer, 'kernel_regularizer'):
    #        layer.kernel_regularizer= regularizers.l2(weight_decay)
    #otolitt.load_weights('../tmp_oto2/log_300_steps_600_600_inception/4000_steps_300_epochs_split_rgb_inceptionV3_600_600_01.h5')
    #for layer in otolitt.layers:
    #    layer.trainable = True
            

    #history_callback = otolitt.model.fit_generator(train_generator,
    #         steps_per_epoch=1600,#4000
    #         epochs=150,
    #         #callbacks=[early_stopper, tensorboard, checkpointer],
    #         callbacks=[early_stopper],
    #        validation_data=(rb_imgs_val_rescaled, np.array(age_val)))

    history_callback = otolitt.fit_generator(train_generator,
            steps_per_epoch=1600,#4000
            epochs=150,
            #callbacks=[early_stopper, tensorboard, checkpointer],
            callbacks=[early_stopper],
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

def read_imr(pandas_df, dir_path, id_column, np_images, end_count, list_age):
    global new_shape
    
    found_count=0
    pandas_df['Totalt'] = pandas_df['Totalt'].dropna()
    for i in range(0, len(pandas_df)):
        if not pd.isnull(pandas_df['Totalt'].values[i]):
            if pandas_df['Totalt'].values[i] != None and not pd.isnull(pandas_df[id_column].values[i]):
                id = pandas_df[id_column].values[i]+'.tif'
                path = os.path.join(dir_path, id )
                my_file = Path(path)
                if my_file.is_file() : 
                    pil_img = load_img(path, grayscale=True)
                    smaller_img = pil_img.resize( (new_shape[1], new_shape[0]))
                    np_images[end_count+found_count] = img_to_array(smaller_img)
                    age.append(pandas_df['Totalt'].values[i])
                    found_count += 1
                my_file = None            
    return found_count
    

if __name__ == '__main__':
    do_train()
    print("end")    