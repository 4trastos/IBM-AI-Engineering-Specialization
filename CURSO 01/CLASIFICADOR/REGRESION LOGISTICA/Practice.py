import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import preprocessing

ChurnData = pd.read_csv("ChurnData.csv")
ChurnData.head()

ChurnData = ChurnData[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
ChurnData['churn'] = ChurnData['churn'].astype('int')
ChurnData.head()

ChurnData.shape

X = np.asarray(ChurnData[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]
y = np.asarray(ChurnData[['churn']])
y[0:5]

X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
y_train = y_train.ravel()
y_test = y_test.ravel()
print('Train Set: ', X_train.shape, y_train.shape)
print('Test Set: ', X_test.shape, y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.1, solver='sag').fit(X_train, y_train)
LR

yhat_proba = LR.predict_proba(X_test)
yhat_proba

####################################################
##### EVALUACIÓN DEL MODELO CON log loss ###########

from sklearn.metrics import log_loss
lg_lss = log_loss(y_test, yhat_proba)
print(f"log loss index: {lg_lss}")
print ("LogLoss: : %.2f" % log_loss(y_test, yhat_proba))