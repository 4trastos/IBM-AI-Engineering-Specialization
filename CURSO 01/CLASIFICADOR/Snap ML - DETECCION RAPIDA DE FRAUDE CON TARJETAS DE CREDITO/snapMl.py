# Import the libraries we need to use in this lab
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

raw_data = pd.read_csv('creditcard.csv')				# read the dataset
print ("There are " + str(len(raw_data)) + " observations in the credit card fraud dataset.")
print ("there are " + str(len(raw_data.columns)) + " variables in the dataset.")
raw_data.head()											# display the first rows in the dataset

n_replicas = 10											# Aumentamos x10 la BBDD para que sea mas realista
big_raw_data = pd.DataFrame(np.repeat(raw_data.values, n_replicas, axis=0), columns=raw_data.columns)
print("There are " + str(len(big_raw_data)) + " observations in the inflated credit card fraud dataset.")
print("There are " + str(len(big_raw_data.columns)) + " variables in the dataset.")
big_raw_data.head()

labels = big_raw_data.Class.unique()					# Obtener clases únicas
sizes = big_raw_data.Class.value_counts().values		# Obtener el conteo de cada clase
fix, ax = plt.subplots()								# Gráfica de los recuentos de valores de clase
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
#plt.show()

#### PRACTICE ####
# Las transacciones con tarjeta de crédito tienen importes diferentes. ¿Podrías trazar un histograma que
# muestre la distribución de estos importes? ¿Cuál es el rango de estos importes (mínimo/máximo)?
# ¿Podrías imprimir el porc3entaje del 90% de los valores de los importes?

plt.hist(big_raw_data.Amount.values, 6, histtype='bar', facecolor='g')
#plt.show()
print ("Minimum amount value is ", np.min(big_raw_data.Amount.values))
print ("Maximun amount value is ", np.max(big_raw_data.Amount.values))
print ("90% of the transactions have an amount less or equal than ", np.percentile(big_raw_data.Amount.values, 90))

#### DATASET PREPROCESSING #####
# En esta subsección prepararás los datos para el entrenamiento.

big_raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(big_raw_data.iloc[:, 1:30]) # estandarizar las características eliminando la media y escalando a la varianza unitaria
data_matrix = big_raw_data.values

X = data_matrix[:, 1:30]							# X: matriz de características (para este análisis, excluimos la variable Time del conjunto de datos)
y = data_matrix[:, 30]								# y: labels vector

X = normalize(X, norm='l1')							# data normalization (norma L1. norma de Manhattan o suma de valores absolutos)
													# ajusta los datos para que la suma de los valores absolutos de cada fila sea 1.

# print the shape of the features matrix and the labels vector
print ('X.shape=', X.shape, 'y.shape=', y.shape)

#### Dataset Train/Test Split ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print ('X_train.shape= ', X_train.shape, 'Y_train.shape= ', y_train.shape)
print ('X_test.shape= ', X_test.shape, 'Y_test.shape= ', y_test)


##### Build a Decision Tree Classifier model with Scikit-Learn ####

# (compute_sample_weight) Calcular los pesos de muestra que se utilizarán como entrada para la rutina de entrenamiento de modo que
# tenga en cuenta el desequilibrio de clases presente en este conjunto de datos

w_train = compute_sample_weight('balanced', y_train)
sklearn_dt = DecisionTreeClassifier(max_depth=4, random_state=35)	# para obtener una salida reproducible en múltiples llamadas de función, establezca random_state en un valor entero determinado

# train a Decision Tree Classifier using scikit-learn
t0 = time.time()
sklearn_dt.fit(X_train, y_train, sample_weight=w_train)
sklearn_time = time.time()-t0
print ("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))

##### Build a Decision Tree Classifier model with Snap ML IBM ######

# if not already computed, 
# compute the sample weights to be used as input to the train routine so that 
# it takes into account the class imbalance present in this dataset
# w_train = compute_sample_weight('balanced', y_train)

# import the Decision Tree Classifier Model from Snap ML
from snapml import DecisionTreeClassifier

# Snap ML ofrece entrenamiento de CPU/GPU multiproceso de árboles de decisión, a diferencia de scikit-learn
# para usar la GPU, configure el parámetro use_gpu en True
# snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, use_gpu=True)

# para configurar la cantidad de subprocesos de CPU utilizados en el momento del entrenamiento, configure el parámetro n_jobs
# para obtener una salida reproducible en múltiples llamadas de función, configure random_state en un valor entero determinado

snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, use_gpu=False)
t0 = time.time()
snapml_dt.fit(X_train, y_train, sample_weight=w_train)
snapml_time = time.time()-t0
print ("[Snap ML] Training time (s):  {0:.5f}".format(snapml_time))

#### Evaluate the ScikitLearn and Snap ML Decision Tree Classifier Models ####

training_speedup = sklearn_time/snapml_time				# Snap ML vs Scikit-Learn training speedup
print ('[Decision Tree Classifier] Snap ML vs. Scikit-Learn speedup : {0:.2f}x '.format(training_speedup))
sklearn_pred = sklearn_dt.predict_proba(X_test)[:,1]	# run inference and compute the probabilities of the test samples
sklearn_roc_auc = roc_auc_score(y_test, sklearn_pred)	# evaluate the Compute Area Under the Receiver Operating Characteristic
print('[Scikit-Learn] ROC-AUC score : {0:.3f}'.format(sklearn_roc_auc))

snapml_pred = snapml_dt.predict_proba(X_test)[:,1]
snapml_roc_auc = roc_auc_score(y_test, snapml_pred)   
print('[Snap ML] ROC-AUC score : {0:.3f}'.format(snapml_roc_auc))

#### Construya un modelo de máquina de vectores de soporte con Scikit-Learn ####

from sklearn.svm import LinearSVC
sklearn_svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

# train a linear Support Vector Machine model using Scikit-Learn
t0 = time.time()
sklearn_svm.fit(X_train, y_train)
sklearn_time = time.time()-t0
print ("[Scikit-Learn] Training time (s):  {0:.2f}".format(sklearn_time))

#### Construya un modelo de máquina de vectores de soporte con Snap ML ####

from snapml import SupportVectorMachine
snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False)

t0 = time.time()
snapml_svm.fit(X_train, y_train)
snapml_time = time.time()-t0
print ("[Snap ML] Training time (s):  {0:.2f}".format(snapml_time))

#### Evaluate the Scikit-Learn and Snap ML Support Vector Machine Models ####

training_speedup = sklearn_time/snapml_time
print('[Support Vector Machine] Snap ML vs. Scikit-Learn training speedup : {0:.2f}x '.format(training_speedup))
sklearn_pred = sklearn_svm.decision_function(X_test)
# evaluate accuracy on test set
acc_sklearn = roc_auc_score(y_test, sklearn_pred)
print("[Scikit-Learn] ROC-AUC score:   {0:.3f}".format(acc_sklearn))

snapml_pred = snapml_svm.decision_function(X_test)
# evaluate accuracy on test set
acc_snapml = roc_auc_score(y_test, snapml_pred)
print("[Snap ML] ROC-AUC score:   {0:.3f}".format(acc_snapml))


##### PRACTICE ######

sklearn_pred = sklearn_svm.decision_function(X_test)
snapml_pred = snapml_svm.decision_function(X_test)

from sklearn.metrics import hinge_loss
# evaluate the hinge loss from the predictions
loss_snapml = hinge_loss(y_test, snapml_pred)
print("[Snap ML] Hinge loss:   {0:.3f}".format(loss_snapml))

loss_sklearn = hinge_loss(y_test, sklearn_pred)
print("[Scikit-Learn] Hinge loss:   {0:.3f}".format(loss_snapml))
