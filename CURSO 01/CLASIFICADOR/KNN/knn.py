import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np

df = pd.read_csv('teleCust1000t.csv')
df.head()
# print (fd.head(6))

#### Veamos cuántos clientes de cada clase hay en nuestro conjunto de datos
#### 281 Plus Service, 266 Basic-service, 236 Total Service, and 217 E-Service customers
df['custcat'].value_counts()

df.hist(column='income', bins=50) # Crear el histograma
#plt.show()                        # Mostrar el gráfico

df.columns
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values #.astype(float)
X[0:5]

y = df[['custcat']].values
y[0:5]

## Normalize Data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

### Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)

### Classification
## K nearest neighbor (KNN)

from sklearn.neighbors import KNeighborsClassifier

### Trainig Model
k = 4
neigh = KNeighborsClassifier(n_neighbors= k).fit(X_train, y_train)
neigh

### Predicción con el modelo de prueba (Test Split)
y_hat = neigh.predict(X_test)
y_hat[0:5]

### Evaluación de precisión
from sklearn import metrics
print ('Train set Accuracy: ', metrics.accuracy_score(y_train, neigh.predict(X_train)))
print ('Test set Accuracy: ', metrics.accuracy_score(y_test, y_hat))

#### ¿Qué sucede con otros K?

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1, Ks):

    #Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
    y_hat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, y_hat)

    std_acc[n - 1] = np.std(y_hat == y_test) / np.sqrt(y_hat.shape[0])

mean_acc

#### Grafique la precisión del modelo para un número diferente de vecinos.

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Precisión ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Precisión ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "La mejor precisión fue con", mean_acc.max(), "con k=", mean_acc.argmax()+1) 