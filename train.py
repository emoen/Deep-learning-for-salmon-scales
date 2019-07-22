import pandas as pd
import os
from keras.preprocessing.image import img_to_array, load_img
from pathlib import Path
import math
import numpy as np

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

from keras.utils import to_categorical

from sklearn import preprocessing

from mse_missing_values import missing_mse, missing_mse2
from clean_y_true import read_and_clean_csv_files

def read_img_to_array_and_age_to_df( pandas_df, img_dir, tf_images, end_count):
    global new_shape, id_column

    dir_path = os.path.join(base_dir, img_dir)
    prediktor = pd.DataFrame({}, columns=['sjø', 'smolt', 'smolt_sjø', 'gytarar', 'vill'])

    found_count=0
    pandas_df.gytarar = pandas_df.gytarar.astype('str')
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
            gytar = np.array([1, 0], dtype=np.float32)
            if pandas_df['gytarar'].values[i] == 'nan' or pandas_df['gytarar'].values[i] == '':
                gytar = np.array([0, 1], dtype=np.float32)
            oppdrett = pandas_df.vill.values[i]
            a_pred = pd.Series({'smolt': smolt, 'sjø': sjo, 'smolt_sjø':np.array([smolt, sjo], dtype=np.float32), 'gytarar': gytar, 'vill': oppdrett},name=found_count)
            prediktor = prediktor.append( a_pred )
            found_count += 1
        my_file = None

    return found_count, prediktor

new_shape = (299, 299, 1)
base_dir = '/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param'
id_column = 'ID nr.'
tensorboard_path = './tensorboard_missing_loss'
checkpoint_path = './checkpoints_missing_loss/salmon_scale_inception.{epoch:03d}-{val_loss:.2f}.hdf5'
age = []
dataset_size_smolt_sjo = 9073
dataset_size_smolt = 6246
dataset_size_sjo = 9073
dataset_size_gytar = 9073
dataset_size_oppdrett = 999 #9073

dataset_size_selected = dataset_size_oppdrett
to_predict = 'oppdrett'

#Total number of images in directory 9636
def do_train():
    global new_shape, age, tensorboard_path, checkpoint_path, to_predict, dataset_size_selected

    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    a_batch_size = 32
    add_count = 0

    rb_imgs = np.empty(shape=(999,)+new_shape) #9073,)+new_shape)
    dataset_imgs_selected = np.empty(shape=(dataset_size_selected,)+new_shape)
    print("rb_imgs:"+str(rb_imgs.shape))
    print("dataset_imgs_selected:"+str(dataset_imgs_selected.shape))
    d2015, d2016, d2017, d2018, d2016rb, d2017rb = read_and_clean_csv_files( base_dir )

    add_count, pred15 = read_img_to_array_and_age_to_df(d2015, 'hi2015_in_excel', rb_imgs, add_count)
    #add_count, pred16 = read_img_to_array_and_age_to_df(d2016, 'hi2016_in_excel', rb_imgs, add_count)
    #add_count, pred17 = read_img_to_array_and_age_to_df(d2017, 'hi2017_in_excel', rb_imgs, add_count)
    #add_count, pred18 = read_img_to_array_and_age_to_df(d2018, 'hi2018_in_excel', rb_imgs, add_count)
    #add_count, pred16rb = read_img_to_array_and_age_to_df(d2016rb, 'rb2016', rb_imgs, add_count)
    #add_count, pred17rb = read_img_to_array_and_age_to_df(d2017rb, 'rb2017', rb_imgs, add_count)

    all = pd.DataFrame({}, columns=pred15.columns.values)
    #all = pd.concat([pred15, pred16, pred17, pred18, pred16rb, pred17rb], axis=0, join='outer', ignore_index=False)
    all = pd.concat([pred15], axis=0, join='outer', ignore_index=False)

    #assert (len(all) == len(pred15)+len(pred16)+len(pred17)+len(pred18)+len(pred16rb)+len(pred17rb))
    assert len(all) == len(rb_imgs)
    print("len(all)"+str(len(all)))
    print("len(rb_imgs))"+str(len(rb_imgs)))

    age = np.array([], dtype=np.float32)
    imgs_add_counter = 0
    if to_predict == 'smolt':
        imgs_add_counter = 0
        for i in range(0, len(all)):
            if all.smolt.values[i] != -1.0:
                age = np.append(age, all.smolt.values[i])
                dataset_imgs_selected[imgs_add_counter] = rb_imgs[i]
                imgs_add_counter += 1
    if to_predict == 'sjø':
        for i in range(0, len(all)):
            if all.sjø.values[i] != -1.0:
                age = np.append(age, all.sjø.values[i])
                dataset_imgs_selected[imgs_add_counter] = rb_imgs[i]
                imgs_add_counter += 1
    if to_predict == 'smolt_sjo':
        for i in range(0, len(all)):
            if all.smolt.values[i] != -1.0 or all.sjø.values[i] != -1.0:
                age = np.append(age, all.smolt_sjø.values[i])
                dataset_imgs_selected[imgs_add_counter] = rb_imgs[i]
                imgs_add_counter += 1
    if to_predict == 'gytarar':
        age = np.vstack(all.gytarar.values)
        dataset_imgs_selected = rb_imgs
    if to_predict == 'oppdrett':
        label_encoder = preprocessing.LabelEncoder()
        integer_encoded = label_encoder.fit_transform(all.vill.values)
        encoded = to_categorical(integer_encoded )
        age = encoded
        print("age encoded:**************************")
        print(str(age[0:5]))
        print(str(set(all.vill.values)))
        dataset_imgs_selected = rb_imgs

    print("dataset_imgs_selected.shape"+str(dataset_imgs_selected.shape))
    print("len(dataset_imgs_selected)"+str(len(dataset_imgs_selected)))

    assert len(age) == len(dataset_imgs_selected)

    print("training set size:"+str( add_count ))

    train_set = pd.DataFrame(columns=['img', 'age'])

    train_set['img'] = pd.Series( (v[0] for v in rb_imgs) )
    for i in range(0,len(age)):
        train_set['age'].values[i] = age[i]

    print("rb_imgs:"+str(len(rb_imgs)))
    print("age:"+str(len(age)))

    train_idx, val_idx, test_idx = train_validate_test_split( range(0, len(rb_imgs)) )
    if to_predict == 'gytarar':
        train_idx, val_idx, test_idx = train_validate_test_split(range(0, len(rb_imgs)), validation_set_size = 0.40, test_set_size = 0.05, a_seed = 8)

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

    print("******* gytarar train/valid/test set ************")
    tuples_val = [tuple(l) for l in age_val]
    tuples_train = [tuple(l) for l in age_train]
    tuples_test  = [tuple(l) for l in age_val]
    print("age_train is:"+str(set(tuples_train))+" count (0,1), (1,0):"+str(tuples_train.count((0,1)))+" "+str(tuples_train.count((1,0))))
    print("age_validation is:"+str(set(tuples_val))+" count (0,1), (1,0):"+str(tuples_val.count((0,1)))+" "+str(tuples_val.count((1,0))))
    print("age_test is:"+str(set(tuples_test))+" count (0,1), (1,0):"+str(tuples_test.count((0,1)))+" "+str(tuples_test.count((1,0))))
    print("******* end ************")

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

    gray_model, gray_model_weights = create_inceptionV3_grayscale()
    gray_model = get_fresh_weights( gray_model, gray_model_weights )
    #z = dense2_linear_output(gray_model)
    z = dense3_vill( gray_model )

    scales = Model(inputs=gray_model.input, outputs=[z])
    learning_rate=0.01   #0.0004
    adam = optimizers.Adam(lr=learning_rate)

    #inception = baseline_model4()
    #z = inception.output
    #scales = Model(inputs=inception.input, outputs=z)
    scales.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=[matthews_correlation, 'accuracy' ] ) #, 'mse', 'mape'] )
    for layer in scales.layers:
        layer.trainable = True

    tensorboard, checkpointer = get_checkpoint_tensorboard(tensorboard_path, checkpoint_path)

    print("len(rb_imgs_val_rescaled):"+str(len(rb_imgs_val_rescaled)))
    print("len(age_val):"+str(len(age_val)))
    print("shape age_val:"+str(age_val[0:5]))
    print("age type:"+str(age_val.dtype))

    history_callback = scales.fit_generator(train_generator,
            steps_per_epoch=1600,
            epochs=150,
            callbacks=[early_stopper, tensorboard, checkpointer],
            validation_data=(rb_imgs_val_rescaled,  age_val))

    test_metrics = scales.evaluate(x=rb_imgs_test_rescaled, y=age_test) #, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None)
    test_predictions = scales.predict(rb_imgs_test_rescaled ) #, batch_size=None, verbose=0, steps=None, callbacks=None)
    print("test predictions:")
    print(test_predictions)
    print("test_metrics:")
    print(test_metrics)


def get_checkpoint_tensorboard(tensorboard_path, checkpoint_path):

    tensorboard = TensorBoard(log_dir=tensorboard_path)
    checkpointer = ModelCheckpoint(
        filepath = checkpoint_path,
        verbose = 1,
        save_best_only = True,
        save_weights_only = False)
    return tensorboard, checkpointer

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

    return gray_model, gray_model_weights

#It seems updating a model also updates the wegihts of the previous model. So, get fresh weights.
def get_fresh_weights(gray_model, gray_model_weights):
    gray_model.set_weights(gray_model_weights)
    return gray_model

def base_output(gray_model):
    z = gray_model.output
    z = GlobalMaxPooling2D()(z)
    z = Dense(1024)(z)
    z = Activation('relu')(z)
    return z

def dense3_vill(gray_model):
    z = base_output(gray_model)
    z = Dense(3, input_dim=1024)(z)
    z = Activation('softmax')(z)
    return z

def dense2_gytarar(gray_model):
    z = base_output(gray_model)
    z = Dense(2, input_dim=1024)(z)
    z = Activation('softmax')(z)
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

def nll1(y_true, y_pred):
    """ Negative log likelihood. """

    # keras.losses.binary_crossentropy give the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

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

def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

if __name__ == '__main__':
    do_train()