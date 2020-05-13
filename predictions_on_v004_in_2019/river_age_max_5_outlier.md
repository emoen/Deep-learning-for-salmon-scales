```
outlier3 = abs(y_pred_test - test_age)
>>> outlier3.argsort()[-5:]
array([ 14, 377, 746, 512, 522])
>>> outlier3[14]
2.0608954429626465
>>> outlier3[377]
2.065356969833374
>>> outlier3[746]
2.083889961242676
>>> outlier3[512]
2.176234722137451
>>> outlier3[522]
2.359718084335327

>>> test_age_names[14]
PosixPath('/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param/hi2016_in_excel/Daleelv i Ho_Sportsfiske_2016_FF-klypte004.jpg')
>>> test_age_names[377]
PosixPath('/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param/hi2018_in_excel/da-18-103.jpg')
>>> test_age_names[746]
PosixPath('/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param/rb2017/Eid2017-414.jpg')
>>> test_age_names[512]
PosixPath('/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param/rb2016/Eidfjord2016-117.jpg')
>>> test_age_names[522]
PosixPath('/gpfs/gpfs0/deep/data/salmon-scales/dataset_5_param/hi2015_in_excel/Langfjordelva_Høstfiske_2015_015.jpg')
```
