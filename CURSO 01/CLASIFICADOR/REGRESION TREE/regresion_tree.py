import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Cargar los datos
churn_df = pd.read_csv("ChurnData.csv")
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

# Definir variables independientes (X) y dependientes (y)
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

# Normalizar los datos
X = preprocessing.StandardScaler().fit(X).transform(X)

# Dividir los datos en conjunto de entrenamiento y de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Modelo de Regresión Logística
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.1, solver='liblinear').fit(X_train, y_train)
LR

# Predicción
yhat = LR.predict(X_test)
yhat

yhat_prob = LR.predict_proba(X_test)
yhat_prob

# Evaluar con matriz de confusión y jaccard_score
from sklearn.metrics import jaccard_score, confusion_matrix
jaccard = jaccard_score(y_test, yhat, pos_label=0)
print(f"Jaccard Score: {jaccard}")

# Función para graficar matriz de confusión
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Graficar la matriz de confusión
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, yhat, labels=[1, 0])

# Llamar a la función para graficar la matriz de confusión
plot_confusion_matrix(cm, classes=[1, 0], title='Confusion Matrix')

# Mostrar la gráfica
plt.show()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

print (classification_report(y_test, yhat))

### evaluacion log loss ####

from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)
print (log_loss(y_test, yhat_prob))

###########################################
###########################################
#############    PRACTICE    ##############
###########################################

# Modelo de Regresión Logística
from sklearn.linear_model import LogisticRegression
LR2 = LogisticRegression(C=0.1, solver='sag').fit(X_train, y_train)
y_hat2Proba = LR2.predict_log_proba(X_test)
print ("LogLoss: : %.2f" % log_loss(y_test, y_hat2Proba))