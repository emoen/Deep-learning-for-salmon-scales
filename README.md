# salmon-scale

Comparison of different prediktors. Also added results from prediction of Greenland otolith for comparison

(MAPE: Mean absolute percentage error)
(MCC: mathews correlation coefficient)
Test-set of otoliths and validation set on salmon scales

| Species            | Predict    | MSE  | MAPE | ACC | MCC | training size
| -------------------| -----------|------|------|-----|-----|----------------|
| Greenland Halibut  | age        | 2.65 |0.124 |0.262|x    |8875|
| Salom              | Sea age    |0.239 |0.141 |     |x    |9000|
| Salom              | river age  |0.431 |0.252 |     |x    |6300|
| Salom missing_loss1| river & sea|1.96  |0.542 |0.628|x    |9073|
| Salom missing_loss2| river & sea|2.04  |0.751 |0.566|x    |9073|
| Salom              | Spawned    |x     |x     |     |     | |
| Salom              | Wild/farmed|x     |x     |     |     | |


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
