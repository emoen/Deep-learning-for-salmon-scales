import pandas as pd
import numpy as np
import os

def read_csv(base_dir):
    d2015 = pd.read_csv(os.path.join(base_dir, '2015_5_param_edit.csv'))
    d2016 = pd.read_csv(os.path.join(base_dir, '2016_5_param_edit.csv'))
    d2017 = pd.read_csv(os.path.join(base_dir, '2017_5_param_edit.csv'))
    d2018 = pd.read_csv(os.path.join(base_dir, '2018_5_param_edit.csv'))
    d2016rb  = pd.read_csv(os.path.join(base_dir, 'rb2016_5_param_edit.csv'))
    d2017rb  = pd.read_csv(os.path.join(base_dir, 'rb2017_5_param_edit.csv'))
    print("excel length:"+str(len(d2015)+len(d2016)+len(d2017)+len(d2018)+len(d2016rb)+len(d2017rb)))
    return d2015,d2016,d2017,d2018,d2016rb,d2017rb

def clean_gytarar(d2015, d2016, d2017, d2018, d2016rb, d2017rb):
    d2015.gytarar =pd.Series([False if pd.isnull(f) else True for f in d2015.gytarar.values], index=d2015.index )
    d2016.gytarar =pd.Series([False if pd.isnull(f) else True for f in d2016.gytarar.values], index=d2016.index )
    d2017.gytarar =pd.Series([False if pd.isnull(f) else True for f in d2017.gytarar.values], index=d2017.index )
    d2018.gytarar =pd.Series([False if pd.isnull(f) else True for f in d2018.gytarar.values], index=d2018.index )
    d2016rb.gytarar =pd.Series([False if pd.isnull(f) else True for f in d2016rb.gytarar.values], index=d2016rb.index )
    d2017rb.gytarar =pd.Series([False if pd.isnull(f) else True for f in d2017rb.gytarar.values], index=d2017rb.index )
    return d2015, d2016, d2017, d2018, d2016rb, d2017rb

def clean_sea(d2015, d2016, d2017, d2018, d2016rb, d2017rb):

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
    return d2015, d2016, d2017, d2018, d2016rb, d2017rb

def clean_smolt(d2015, d2016, d2017, d2018, d2016rb, d2017rb):
    d2015.smolt = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2015.smolt] )
    d2016.smolt = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2016.smolt] )
    d2017.smolt = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2017.smolt] )
    d2018.smolt = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2018.smolt] )
    d2016rb.smolt = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2016rb.smolt] )
    d2017rb.smolt = pd.Series( [-1.0 if (f == 0 or np.isnan(f)) else f for f in d2017rb.smolt] )
    return d2015, d2016, d2017, d2018, d2016rb, d2017rb

def clean_farmed_salmon(d2015, d2016, d2017, d2018, d2016rb, d2017rb):
    d2015 = d2015.rename(index=str, columns={'vill/oppdrett': 'vill'})
    d2015.vill = d2015.vill.astype('str')
    d2015.at[d2015['vill']=='Vill', 'vill'] = 'vill'
    d2015.at[d2015['vill']=='Oppdrett', 'vill'] = 'oppdrett'
    d2015.at[d2015['vill']=='.', 'vill'] = 'ukjent'
    d2015.at[d2015['vill']=='nan', 'vill'] = 'ukjent'
    d2015.at[d2015['vill']=='Regnbueørret', 'vill'] = 'ukjent'
    d2015.at[d2015['vill']=='Utsatt', 'vill'] = 'ukjent'

    d2016 = d2016.rename(index=str, columns={'vill/oppdrett': 'vill'})
    d2016.vill = d2016.vill.astype('str')
    d2016.at[d2016['vill']=='Vill', 'vill'] = 'vill'
    d2016.at[d2016['vill']=='Vill ', 'vill'] = 'vill'
    d2016.at[d2016['vill']=='Oppdrett ', 'vill'] = 'oppdrett'
    d2016.at[d2016['vill']=='Oppdrett', 'vill'] = 'oppdrett'
    d2016.at[d2016['vill']=='.', 'vill'] = 'ukjent'
    d2016.at[d2016['vill']=='Sjøørret', 'vill'] = 'ukjent'
    d2016.at[d2016['vill']=='nan', 'vill'] = 'ukjent'
    d2016.at[d2016['vill']=='Utsatt', 'vill'] = 'ukjent'

    d2017 = d2017.rename(index=str, columns={'vill/oppdrett': 'vill'})
    d2017.vill = d2017.vill.astype('str')
    d2017.at[d2017['vill']=='Vill', 'vill'] = 'vill'
    d2017.at[d2017['vill']=='Oppdrett', 'vill'] = 'oppdrett'
    d2017.at[d2017['vill']=='Ikke lesbar', 'vill'] = 'ukjent'
    d2017.at[d2017['vill']=='.', 'vill'] = 'ukjent'
    d2017.at[d2017['vill']=='Utsatt', 'vill'] = 'ukjent'

    d2018 = d2018.rename(index=str, columns={'vill/oppdrett': 'vill'})
    d2018.vill = d2018.vill.astype('str')
    d2018.at[d2018['vill']=='Vill', 'vill'] = 'vill'
    d2018.at[d2018['vill']=='Oppdrett', 'vill'] = 'oppdrett'
    d2018.at[d2018['vill']=='nan', 'vill'] = 'ukjent'
    d2018.at[d2018['vill']=='Mangler skjell', 'vill'] = 'ukjent'
    d2018.at[d2018['vill']=='Ikkje lesbar', 'vill'] = 'ukjent'
    d2018.at[d2018['vill']=='Sjøaure', 'vill'] = 'ukjent'
    d2018.at[d2018['vill']=='Mangler skjellprøve', 'vill'] = 'ukjent'
    d2018.at[d2018['vill']=='Ikke lesbar', 'vill'] = 'ukjent'
    d2018.at[d2018['vill']=='Skjell Mangler', 'vill'] = 'ukjent'
    d2018.at[d2018['vill']=='Mangler Skjell', 'vill'] = 'ukjent'
    d2018.at[d2018['vill']=='Ikke lesbart', 'vill'] = 'ukjent'
    d2018.at[d2018['vill']=='Utsatt', 'vill'] = 'ukjent'
    d2018.at[d2018['vill']=='.', 'vill'] = 'ukjent'
    d2018.at[d2018['vill']=='Manglar skjell', 'vill'] = 'ukjent'
    d2018.at[d2018['vill']=='Sjøørret', 'vill'] = 'ukjent'
    d2018.at[d2018['vill']=='Ørret', 'vill'] = 'ukjent'

    d2016rb = d2016rb.rename(index=str, columns={'vill/oppdrett': 'vill'})
    d2016rb.vill = d2016rb.vill.astype('str')
    d2016rb.at[d2016rb['vill']=='Vill', 'vill'] = 'vill'
    d2016rb.at[d2016rb['vill']=='Oppdrett', 'vill'] = 'oppdrett'
    d2016rb.at[d2016rb['vill']=='?', 'vill'] = 'ukjent'
    d2016rb.at[d2016rb['vill']=='.', 'vill'] = 'ukjent'
    d2016rb.at[d2016rb['vill']=='nan', 'vill'] = 'ukjent'

    d2017rb = d2017rb.rename(index=str, columns={'vill/oppdrett': 'vill'})
    d2017rb.vill = d2017rb.vill.astype('str')
    d2017rb.at[d2017rb['vill']=='Vill', 'vill'] = 'vill'
    d2017rb.at[d2017rb['vill']=='Oppdrett', 'vill'] = 'oppdrett'
    d2017rb.at[d2017rb['vill']=='?', 'vill'] = 'ukjent'
    d2017rb.at[d2017rb['vill']=='.', 'vill'] = 'ukjent'
    d2017rb.at[d2017rb['vill']=='nan', 'vill'] = 'ukjent'
    return d2015, d2016, d2017, d2018, d2016rb, d2017rb

def read_and_clean_4_param_csv( base_dir ):
    d2015,d2016,d2017,d2018,d2016rb,d2017rb = read_csv(base_dir)
    d2015,d2016,d2017,d2018,d2016rb,d2017rb = clean_sea(d2015, d2016, d2017, d2018, d2016rb, d2017rb)
    d2015,d2016,d2017,d2018,d2016rb,d2017rb = clean_smolt(d2015, d2016, d2017, d2018, d2016rb, d2017rb)
    d2015,d2016,d2017,d2018,d2016rb,d2017rb = clean_farmed_salmon(d2015, d2016, d2017, d2018, d2016rb, d2017rb)
    d2015,d2016,d2017,d2018,d2016rb,d2017rb = clean_gytarar(d2015, d2016, d2017, d2018, d2016rb, d2017rb)
    return d2015, d2016, d2017, d2018, d2016rb, d2017rb
