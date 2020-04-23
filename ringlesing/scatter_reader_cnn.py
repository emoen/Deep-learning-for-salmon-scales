import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

cnn_n_ring = pd.read_csv('sea_age_prediction_ringlesing2020.csv', sep=' ')
x = cnn_n_ring['y_true']
x = x[0:149]
y = cnn_n_ring['y_hat']
y = y[0:149]
mse = mean_squared_error(x, y) #mse=mse = mean_squared_error(x, y)

plt.plot(x, y, 'o', color='black');
plt.plot(x, x)
plt.show()