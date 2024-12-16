from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
import time
import warnings
import gc, sys
warnings.filterwarnings('ignore')

raw_data = pd.read_csv("yellow_tripdata_2019-06.csv")
print ("There are: " + str(len(raw_data)) + " observations in the dataset.")
print ("There are: " + str(len(raw_data.columns)) + " variables in the dataset.")

raw_data.head()

###############################################################
##################### ANALIZANDO EL DATASET ###################
###############################################################

#Reducing the data size to 100000 records
raw_data = raw_data.head(10000)

raw_data = raw_data[raw_data['tip_amount'] > 0]
raw_data = raw_data[(raw_data['tip_amount'] <= raw_data['fare_amount'])]
raw_data = raw_data[(raw_data['fare_amount'] >= 2) & (raw_data['fare_amount'] < 200)]
clean_data = raw_data.drop(['total_amount'], axis=1)

del raw_data
gc.collect

print ("There are: " + str(len(clean_data)) + " observations in the dataset.")
print ("There are: " + str(len(clean_data.columns)) + " variables in the dataset.")

plt.hist(clean_data.tip_amount.values, 16, histtype='bar', facecolor='g')
plt.show()

print ("Minimum amount value is: ", np.min(clean_data.tip_amount.values))
print ("Maximum amount value is:", np.max(clean_data.tip_amount.values))
print ("90% of the trips have a tip amount less or equal than ", np.percentile(clean_data.tip_amount.values, 90))

clean_data.head()

###############################################################
################## PREPROCESSING EL DATASET ###################
###############################################################

clean_data['tpep_dropoff_datetime'] = pd.to_datetime(clean_data['tpep_dropoff_datetime'])
clean_data['tpep_pickup_datetime'] = pd.to_datetime(clean_data['tpep_pickup_datetime'])
clean_data['pickup_hour'] = clean_data['tpep_pickup_datetime'].dt.hour
clean_data['dropoff_hour'] = clean_data['tpep_dropoff_datetime'].dt.hour
clean_data['pickup_day'] = clean_data['tpep_pickup_datetime'].dt.weekday
clean_data['dropoff_day'] = clean_data['tpep_dropoff_datetime'].dt.weekday
clean_data['trip_time'] = (clean_data['tpep_dropoff_datetime'] - clean_data['tpep_pickup_datetime']).dt.total_seconds() / 60

first_n_rows = 1000000
clean_data = clean_data.head(first_n_rows)

clean_data = clean_data.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)

get_dummy_col = ["VendorID","RatecodeID","store_and_fwd_flag","PULocationID", "DOLocationID","payment_type", "pickup_hour", "dropoff_hour", "pickup_day", "dropoff_day"]
proc_data = pd.get_dummies(clean_data, columns = get_dummy_col)

del clean_data
gc.collect()

y = proc_data[['tip_amount']].values.astype('float32')
proc_data = proc_data.drop(['tip_amount'], axis=1)
X = proc_data.values
X = normalize(X, axis=1, norm='l1', copy=False)
print ('X.shape=', X.shape, 'y.shape=', y.shape)

###############################################################
################## DATASET train/test SPLIT ###################
###############################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('X_train.shape=', X_train.shape, 'Y_train.shape=', y_train.shape)
print('X_test.shape=', X_test.shape, 'Y_test.shape=', y_test.shape)

###############################################################
######## MODEL DECISION TREE REGRESSOR SCIKIT-LEARN ###########
###############################################################

from sklearn.tree import DecisionTreeRegressor

sklearn_dt = DecisionTreeRegressor(max_depth=8, random_state=35)
t0 = time.time()
sklearn_dt.fit(X_train, y_train)
sklearn_time = time.time() - t0
print("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))

###############################################################
########### MODEL DECISION TREE REGRESSOR SNAP ML #############
###############################################################

from snapml import DecisionTreeRegressor

snapml_dt = DecisionTreeRegressor(max_depth=8, random_state=45, n_jobs=4)
t0 = time.time()
snapml_dt.fit(X_train, y_train)
snapml_time = time.time() - t0
print("[Snap ML] Training time (s):  {0:.5f}".format(snapml_time))

###############################################################
########### Evaluate the Scikit-Learn and Snap ML #############
###############################################################

training_speedup = sklearn_time / snapml_time
print ('[Decision Tree Regressor] Snap ML vs. Scikit-Learn speedup : {0:.2f}x '.format(training_speedup))

sklearn_pred = sklearn_dt.predict(X_test)

sklearn_mse = mean_squared_error(y_test, sklearn_pred)
print('[Scikit-Learn] MSE score : {0:.3f}'.format(sklearn_mse))

snapml_pred = snapml_dt.predict(X_test)

snapml_mse = mean_squared_error(y_test, snapml_pred)
print('[Snap ML] MSE score : {0:.3f}'.format(snapml_mse))