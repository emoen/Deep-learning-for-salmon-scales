#Korsbrekk2016-161.jpg JPEG 2048x1536 2048x1536+0+0 8-bit DirectClass 906KB 0.000u 0:00.000
#131.jpg JPEG 2560x1920 2560x1920+0+0 8-bit DirectClass 741KB 0.000u 0:00.000

#/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param/rundlesing2020

#/gpfs/gpfs0/deep/projects/em-salmon-scales/checkpoints_river_28_april_2020_v1.1.0
#/gpfs/gpfs0/deep/projects/em-salmon-scales/tensorboard_river_28_april_2020_v1.1.0


import numpy as np
import pandas as pd
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
import scipy
import tensorflow as tf
from keras.models import load_model

from clean_y_true import read_and_clean_4_param_csv
from train_util import read_images, load_xy, get_checkpoint_tensorboard, create_model_grayscale, get_fresh_weights, base_output, dense1_linear_output, train_validate_test_split
import efficientnet.keras as efn

def test_sea_predictions():
    new_shape = (380, 380, 3)
    IMG_SHAPE = (380, 380)

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    project_root_dir = '/gpfs/gpfs0/deep/projects/em-salmon-scales/'
    model_path= project_root_dir+'checkpoints_river_28_april_2020_v1.1.0/salmon_scale_efficientnetB4.055-0.30.hdf5'
    model_pred_path = project_root_dir+'tensorboard_river_28_april_2020_v1.1.0/y_pred_river.txt'
    ringlesing_y_true = project_root_dir+'ringlesing2020_pred_smolt.csv'

    ringlesing_path = "/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param/rundlesing2020" #fra Åse
    #ringlesing_path = "/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param/ringlesning2019" # using gdal_translate
    max_dataset_size =  len([name for name in os.listdir(ringlesing_path) if os.path.isfile(os.path.join(ringlesing_path, name))])

    ringlesing_imgs = np.empty(shape=(max_dataset_size,)+new_shape)
    ringlesing_imgs, filename = read_images_from_ringlesing(ringlesing_path, ringlesing_imgs, IMG_SHAPE)
    river_age_model = load_model(model_path)

    ringlesing_imgs = np.multiply(ringlesing_imgs, 1./255)
    y_hat = river_age_model.predict(ringlesing_imgs)

    df_y_hat = pd.DataFrame(columns=['filename','y_hat', 'fishno'])
    df_y_hat['filename'] = filename
    df_y_hat['y_hat'] = y_hat
    df_y_hat['fishno'] = [f[:-4] for f in filename]
    #df_y_hat.to_csv('sea_age_prediction_ringlesing2020.csv', sep=' ', index=False)
    df_y_hat['fishno']=pd.to_numeric(df_y_hat['fishno'])
    df_y_hat = df_y_hat.sort_values(by=['fishno'])

    df_y_true = pd.read_csv(ringlesing_y_true, sep=' ')

    #mse_pred = mean_squared_error(df_y_true['y_true'], df_y_hat['y_hat'])
    #print(mse_pred) # mse=0.06686158509602429  

    np.testing.assert_array_equal(df_y_hat['filename'].values, df_y_true['filename'].values)
    #df_outliers = pd.DataFrame(columns=['fishno','filename','y', 'y_hat', 'magnitude'])
    #df_outliers['filename'] = df_y_true['filename'].values
    #df_outliers['fishno'] = [f[:-4] for f in df_y_true['filename'].values]
    #df_outliers['y'] = df_y_true['y_true']
    #df_outliers['y'] = df_outliers['y'].astype(float) 
    #df_outliers['y_hat'] = df_y_hat['y_hat'].values
    #df_outliers['magnitude'] = np.abs(df_outliers['y'].values-df_outliers['y_hat'].values)
    #df_outliers = df_outliers.sort_values(by=['magnitude'])
    #df_outliers.to_csv('river_age_magnitude_error_ringlesing2020.csv', sep=' ', index=False)

    # test on test_set
    do_test_sea(model_pred_path, new_shape, river_age_model)

def read_images_from_ringlesing(ringlesing_path, rb_imgs, IMG_SHAPE):
    
    found_count=0
    filename=list()
    for image_name in os.listdir(ringlesing_path):
        path = os.path.join(ringlesing_path, image_name )
        pil_img = load_img(path, target_size=IMG_SHAPE, grayscale=False)
        array_img = img_to_array(pil_img, data_format='channels_last')
        rb_imgs[found_count] = array_img
        filename.append(image_name)
        found_count += 1
    
    return rb_imgs, filename
    
def do_test_sea(model_pred_path, new_shape, river_age_model):
    rb_imgs, all_sea_age, all_smolt_age, all_farmed_class, all_spawn_class, all_filenames = load_xy()

    uten_ukjent = len(all_smolt_age) - all_smolt_age.count(-1.0)
    rb_imgs2 = np.empty(shape=(uten_ukjent,)+new_shape)
    unique, counts = np.unique(all_smolt_age, return_counts=True)
    print("age distrib:"+str( dict(zip(unique, counts)) ))

    all_smolt_age2 = []
    all_filenames2 = []
    found_count = 0
    for i in range(0, len(all_smolt_age)):
        if all_smolt_age[i] > 0:
            rb_imgs2[found_count] = rb_imgs[i]
            all_smolt_age2.append(all_smolt_age[i])
            found_count += 1
            all_filenames2.append(all_filenames[i])

    assert found_count == uten_ukjent

    age = all_smolt_age2
    rb_imgs=rb_imgs2

    train_idx, val_idx, test_idx = train_validate_test_split( range(0, len(rb_imgs)) )

    test_rb_imgs = np.empty(shape=(len(test_idx),)+new_shape)
    test_age = []
    test_age_names = []
    for i in range(0, len(test_idx)):
        test_rb_imgs[i] = rb_imgs[test_idx[i]]
        test_age.append(age[test_idx[i]])
        test_age_names.append(all_filenames2[test_idx[i]])
        
    test_rb_imgs = np.multiply(test_rb_imgs, 1./255)    
    y_hat = river_age_model.predict(test_rb_imgs)

    model_y_true_df = pd.read_csv(model_pred_path, sep=' ')
        
    test_age_names_posix_path_as_str= [str(p) for p in test_age_names]
    np.testing.assert_array_equal(test_age_names_posix_path_as_str, model_y_true_df['sea_name'].values)
    np.testing.assert_array_equal(test_age, model_y_true_df['y'].values)
        
    mse_model_y_true = mean_squared_error(model_y_true_df['y'], model_y_true_df['y_hat'])
    mse_pred = mean_squared_error(test_age, y_hat)
    print("compare prediction error while testing on test-set v.s. prediction error after loading model on test-set")
    print("MSE of prediction on test-set while testing:")
    print(mse_model_y_true)
    print("MSE of prediction on test-set after loading model+weights:")
    print(mse_pred)
    print("difference")
    print(str(abs(mse_model_y_true-mse_pred)))
    
    

    
if __name__ == '__main__':
    test_sea_predictions()    