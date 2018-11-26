import pandas as pd
import os
from keras.preprocessing.image import img_to_array, load_img
from pathlib import Path

import xlrd


df  = pd.read_csv('Totalark_2016_Malde.tsv', sep='\t', skiprows=4)
d2014 = pd.read_csv('data_2014.csv')
d2015 = pd.read_csv('data_2015.csv')
d2016 = pd.read_csv('data_2016.csv')

count = 0
count2=0
rb_imgs = []
new_shape = (299, 299, 1)
for i in range(0, len(df) ):
    if df['Totalt'].values[i] != None:
        fylke =  df['Fylke'].values[i]
        #if df['Fylke'].values[i] == 'Hordaland': 
        #    break
        elv = df['Elv'].values[i]+' 2016'
        id = df['ID nr.'].values[i]+'.jpg'
        path = os.path.join('/gpfs/gpfs0/deep/data/salmon-scales/from_RB', fylke, elv, id )
        if  my_file.is_file() : 
            count2 +=1
            pil_img = load_img(path, grayscale=True)
            smaller_img = pil_img.resize( (new_shape[1], new_shape[0]))
            rb_imgs.append(img_to_array(smaller_img))
        my_file = None

count16=0
for i in range(0, len(d2016)):
    if d2016['Totalt'].values[i] != None:
        count16 += 1
        id = d2016['ID nr.'].values[i]+'.tif'
        path = os.path.join(
            '/gpfs/gpfs0/deep/data/salmon-scales/imr-2014..2016/X\ 2016\ X/X\ BILDER\ 2016\ X', id )
        if my_file.is_file() : 
            pil_img = load_img(path, grayscale=True)
            smaller_img = pil_img.resize( (new_shape[1], new_shape[0]))
            rb_imgs.append(img_to_array(smaller_img))
        my_file = None            

count15=0
for i in range(0, len(d2015)):
    if d2015['Totalt'].values[i] != None:
        count15 += 1

count14=0
for i in range(0, len(d2014)):
    if d2014['Totalt'].values[i] != None:
        count14 += 1

num_ex = count2 + count14 + count15 + count16
