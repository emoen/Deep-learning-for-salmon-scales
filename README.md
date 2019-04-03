# salmon-scale

Comparison of different metrics for prediction of salmon scales. I have also added metric from Greenland otolith prediction for comparison. The metrics is from the validation set. Except the first line which is from Greenland Halibut and is calculated from mean of pairs of right and left otolith.<br />

(MAPE: Mean absolute percentage error)<br />
(MCC: mathews correlation coefficient)<br />

| Species             | Predict    |val_LOSS| MSE  | MAPE | ACC | MCC |#trained |activ. f| class imb. pos ex |
| --------------------| -----------|--------|------|------|-----|-----|---------|--------|-------------------|
| Greenland Halibut(1)| age        | x      |2.65  |0.124 |0.262|x    |8875     | linear | x | 
| Greenland Halibut(2)| age        | -"-    |2.82  |0.136 |0.294|x    |8875     | linear | x |
| Salmon              | sea age    | -"-    |0.239 |0.141 |0.822|x    |ca 9000  | linear | x |
| Salmon              | river age  | -"-    |0.431 |0.252 |0.585|x    |6300     | linear | x |
| Salmon missing_loss1| river & sea|9.4372  |2.955 |0.97  |0.707|x    |9073     | linear | x |
| Salmon missing_loss2| river & sea|0.5915  |2.992 |0.974 |0.707|x    |9073     | linear | x |
| Salmon missing_loss3| river & sea|2.0107  |2.011 |0.744 |0.607|x    |9073     | linear | x |
| Salmon (3)          | Spawned    | 0.393  |x     |x     |0.976|0.951|9073     | softmax| 422 (4.7%) |
| Salmon              | Wild/farmed|x       |x     |x     |     |     |         |        |  |

* (1) is test-set <br/>
* (2) is validation-set <br/>
* (3) Validation set was 40%, test set 5%. 
** Training-set (negative example, positive example): (4861, 129)
** Validation-set (negative example, positive example): (3541 89) - 89/(3541+89)= 0.025, 1-0.25 = 0.975
* missing_loss1 - missing_mse(y_true, y_pred) in https://github.com/emoen/salmon-scale/blob/master/mse_missing_values.py <br />
* missing_loss2 - missing_mse2(y_true, y_pred) in https://github.com/emoen/salmon-scale/blob/master/mse_missing_values.py <br />
* missing_loss3 - classic mse with 2 outputs <br />

Note val_acc is 0.7068 in almost every epoch (except 2. epoch of missing_loss2 training.) <br />

Missing_loss1/2 is same the same network - but with Dense(2, 'linear') so it predicts both sea and river age.
```
>>> df = pd.DataFrame({}, d2015.columns.values)
>>> df = df.append(d2015)
>>> df = df.append(d2016)
>>> df = df.append(d2017)
>>> df = df.append(d2018)
>>> df = df.append(d2016rb)
>>> df = df.append(d2017rb)
>>> len(df)
16601
>>> df.sjø.value_counts()
 2.0     7737
 1.0     3809
 3.0     2832
-1.0     1513
 4.0      486
 5.0      123
 6.0       59
 7.0       22
 8.0        9
 9.0        3
 11.0       1
 12.0       1
Name: sjø, dtype: int64
>>> df.smolt.value_counts()
-1.0    7923
 3.0    4900
 2.0    2937
 4.0     549
 1.0     216
 5.0      62
 6.0       8
Name: smolt, dtype: int64

```
