import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

my_data = pd.read_csv('drug200.csv')
my_data.head() # Muestra las primeras 5 lineas y6 así vemos los nombres de los campos
my_data.shape  # tamaño de la matriz de la base de datos (x, y)

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

# como algunas caracteristicas como **Sex** o **BP** no son numéricas si no que son categóricas. Hay que combertirlas en valores numéricos
# por que Sklearn Decision Trees no maneja variables categóricas

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:,1] = le_sex.transform(X[:,1])

le_bp = preprocessing.LabelEncoder()
le_bp.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_bp.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['HIGH', 'NORMAL'])
X[:,3] = le_Chol.transform(X[:,3])

X[0:5]

# completamos la variable objetivo
y = my_data["Drug"]
y[0:5]

#### CREACIÓN DECISION TREE #####
# Creamos el split test/train importamos la libreria

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

# Imprima la forma de X_train y y_train. Asegúrese de que las dimensiones coincidan.
print ('Shape of X training set {}'.format(X_train.shape), '&', ' Size of Y training set {}'.format(y_train.shape))

# Imprima la forma de X_test y y_test. Asegúrese de que las dimensiones coincidan.
print ('Shape of X test set {}'.format(X_test.shape), '&', 'Shape of y test set {}'.format(y_test.shape))

#### MODELING ####
# Primero crearemos una instancia del clasificador DecisionTreeClassifier denominada DrugTree.
# Dentro del clasificador, especifique criterion="entropy" para que podamos ver la ganancia de información de cada nodo.

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree # it shows the default parameters

# A continuación, ajustaremos los datos con la matriz de características de entrenamiento X_trainset
# y el vector de respuesta de entrenamiento y_trainset

drugTree.fit(X_train, y_train)

#### PREDICTION ####
# Hagamos algunas predicciones en el conjunto de datos de prueba y almacenémoslas en una variable llamada predTree

predTree = drugTree.predict(X_test)

# Puede imprimir predTree y y_test si deseas comparar visualmente las predicciones con los valores reales.
print (predTree [0:5])
print (y_test [0:5])

### EVALUACIÓN ###
# A continuación, importemos metrics de sklearn y verifiquemos la precisión de nuestro modelo.

from sklearn import metrics
print ("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))

### VISUALIZATION ###
# Visualicemos el árbol

from sklearn.tree import export_graphviz
import subprocess  # para ejecutar fuera de Jupyter Notebooks
export_graphviz(drugTree, out_file='tree.dot', filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
# !dot -Tpng tree.dot -o tree.png
subprocess.run(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

subprocess.run(['open', 'tree.png'])