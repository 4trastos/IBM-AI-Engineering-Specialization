def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import r2_score

df = pd.read_csv("Weather_Data.csv")
#print(df.head())

df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)

df_sydney_processed.drop('Date',axis=1,inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
y = df_sydney_processed['RainTomorrow']

x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=10)
#print('Train Set: ', x_train.shape, y_train.shape)
#print('Test Set: ', y_train.shape, y_test.shape)

LinearReg = LinearRegression()
LinearReg.fit(x_train, y_train)
#print("Coeficientes: ", LinearReg.coef_)
#print(f"Intercepción: {LinearReg.intercept_:.2f}")

predictions = LinearReg.predict(x_test)

LinearRegression_MAE = np.mean(np.absolute(y_test - predictions))
LinearRegression_MSE = np.mean((y_test - predictions) ** 2)
LinearRegression_R2 = r2_score(y_test, predictions)

# Mostrar el MAE, MSE y R² en un DataFrame
Report = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'R²'],
    'Value': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]
})

print(Report)

KNN = KNeighborsClassifier(n_neighbors=4).fit(x_train, y_train)

predictions = KNN.predict(x_test)

KNN_Accuracy_Score = metrics.accuracy_score(y_test, predictions) 
KNN_JaccardIndex = metrics.jaccard_score(y_test, predictions)
KNN_F1_Score = metrics.f1_score(y_test, predictions)

#print(f"Accuracy: {KNN_Accuracy_Score:.4f}")
#print(f"Jaccard Index: {KNN_JaccardIndex:.4f}")
#print(f"F1 Score: {KNN_F1_Score:.4f}")

Tree = DecisionTreeClassifier()
Tree.fit(x_train, y_train)

predictions = Tree.predict(x_test)

Tree_Accuracy_Score = metrics.accuracy_score(y_test, predictions) 
Tree_JaccardIndex = metrics.jaccard_score(y_test, predictions)
Tree_F1_Score = metrics.f1_score(y_test, predictions)

x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=1)

LR = LogisticRegression(solver='liblinear').fit(x_train, y_train)

predictions = np.asarray(LR.predict(x_test))
predict_proba = np.asarray(LR.predict_proba(x_test))

LR_Accuracy_Score = metrics.accuracy_score(y_test, predictions) 
LR_JaccardIndex = metrics.jaccard_score(y_test, predictions)
LR_F1_Score = metrics.f1_score(y_test, predictions)
LR_Log_Loss = metrics.log_loss(y_test, predict_proba)

SVM = svm.SVC(kernel='rbf')
SVM.fit(x_train, y_train)

predictions = SVM.predict(x_test)
predictions[0:5]

SVM_Accuracy_Score = metrics.accuracy_score(y_test, predictions) 
SVM_JaccardIndex = metrics.jaccard_score(y_test, predictions)
SVM_F1_Score = metrics.f1_score(y_test, predictions)

Report = pd.DataFrame({
    'Metric': ['ACCURACY', 'JACCARD', 'F1'],
    'Value': [SVM_Accuracy_Score, SVM_JaccardIndex, SVM_F1_Score]
})

print(Report)