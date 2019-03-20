# salmon-scale

Comparison of different metrics for prediction of salmon scales. I have also added metric from Greenland otolith prediction for comparison. The metrics is from the validation set. Except the first line which is from Greenland Halibut and is calculated from mean of pairs of right and left otolith.<br />

(MAPE: Mean absolute percentage error)<br />
(MCC: mathews correlation coefficient)<br />

| Species             | Predict    |val_LOSS| MSE  | MAPE | ACC | MCC | training size
| --------------------| -----------|--------|------|------|-----|-----|----------------|
| Greenland Halibut*  | age        | x      |2.65  |0.124 |0.262|x    |8875|
| Greenland Halibut** | age        | -"-    |2.82  |0.136 |0.294|x    |8875|
| Salmon              | sea age    | -"-    |0.239 |0.141 |0.822|x    |ca 9000|
| Salmon              | river age  | -"-    |0.431 |0.252 |0.585|x    |6300|
| Salmon missing_loss1| river & sea|9.4372 |2.955|0.97  |0.707|x    |9073|
| Salmon missing_loss2| river & sea| x      |***  |*** |***  |x    |9073|
| Salmon missing_loss3| river & sea|  |  | ||x    |9073|
| Salmon              | Spawned    |x       |x     |      |     | |
| Salmon              | Wild/farmed|x       |x     |      |     | |

* is test-set <br/>
** is validation-set <br/>
missing_loss1 - missing_mse(y_true, y_pred) in https://github.com/emoen/salmon-scale/blob/master/mse_missing_values.py <br />
missing_loss2 - missing_mse2(y_true, y_pred) in https://github.com/emoen/salmon-scale/blob/master/mse_missing_values.py <br />
missing_loss2 - classic mse with 2 outputs <br />

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
