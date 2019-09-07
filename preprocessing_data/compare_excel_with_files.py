import pandas as pd
from os import listdir
from os.path import isfile, join

#mypath = '/data/delphi/alle/Laks 2014-2015-2016-2017/X 2017 X/X BILDER X 2017'
#mypath = '/data/delphi/alle/Laks 2014-2015-2016-2017/X 2016 X/X BILDER 2016 X'
mypath = '\\salmon_scale\\jpg\\rb_2018\\'

mypath='C:\\salmon_scale\\dataset_5_param\\hi\\2018'
dest = 'C:\\salmon_scale\\dataset_5_param\\hi\\2018_in_excel'

csv_path = 'C:\\salmon_scale\\dataset_5_param\\'

onlyfiles = [f.lower() for f in listdir(mypath) if isfile(join(mypath, f))]
d2018 = pd.read_csv(csv_path+'2018_5_param.csv')
rb2016 = pd.read_csv('rb_2016_from_excel.csv')
rb2017 = pd.read_excel('excel\\rb_2016_alder.xlsx')
#>>> rb2017.values[3625][0].decode('utf8').encode('latin1')
#'Fj\xe6ra2017-002'
for i in range(0, len(rb2016)):
    rb2016.values[i][0] = rb2016.values[i][0].decode('utf8').encode('latin1')

in_excel_and_file = [f+'.jpg' for f in d2017['ID nr.'] if f+'.jpg' in onlyfiles]
in_excel_and_file = [f.lower()+'.jpg' for f in d2018['ID nr.'] if (f.decode('utf8').encode('latin1').lower()+'.jpg' in onlyfiles)] #1143, 1034, 863
in_file_and_excel = [f for f in onlyfiles if f[0:-4] in rb2016['ID nr.'].values ] #1123, 1034, 863

#copy files:
for i in range(0, len(in_excel_and_file)):
    copy2(mypath+'\\'+in_excel_and_file[i].decode('utf8').encode('latin1'), dest)
    
alist = []
for i in range(0, len(onlyfiles)):
    if not onlyfiles[i].decode('latin1').encode('utf8') in in_excel_and_file:
        alist.append(onlyfiles[i])
         copy2(mypath+'\\'+onlyfiles[i], 'C:\\salmon_scale\\dataset_5_param\\hi\\2018_not_in_excel')    
         
############
 
only_in_excel = [f+'.jpg' for f in rb2016['ID nr.'] if not(f.decode('utf8').encode('latin1')+'.jpg' in onlyfiles)] #(15,16,17):123, 55, 231, (rb17):148
only_in_excel_idx = [i for i in range(0, len(rb2016)) if not(rb2016.values[i][0].decode('utf8').encode('latin1')+'.jpg' in onlyfiles)] 
#idx_only_in_excel = [i for i in range(0, len(rb2017)) if rb2017.values[i][0].decode('utf8').encode('latin1')+'.jpg' in only_in_excel]
df_only_in_excel = rb2016.iloc[only_in_excel_idx] #[idx_only_in_excel]
df_only_in_excel.to_csv('rb2016_bare_i_excel.csv', sep=',', index=False, encoding='utf-8')

decoded = [a.decode('utf8') for a in rb2016['ID nr.'].values ]
only_in_file = [f for f in onlyfiles if not(f.decode('latin1')[0:-4] in decoded) ]
adf = pd.DataFrame(columns=['files'])
adf['files'] =  only_in_file
adf.to_csv('rb2016_fil_ikke_excel.csv', index=False)

assert len(set(in_excel_and_file)) == len(in_file_and_excel)
duplicate_in_excel = set([x for x in in_excel_and_file if in_excel_and_file.count(x) > 1])



