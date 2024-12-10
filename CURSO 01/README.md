# Ejemplo de Machine Learning con SVM y Matriz de Confusión

Este script demuestra el uso de una Máquina de Vectores de Soporte (SVM) para clasificación, preprocesamiento de datos, evaluación del modelo usando una matriz de confusión y guardado del modelo entrenado con `pickle`.

## Pasos:

1. **Preprocesamiento de Datos**:
   Los datos se estandarizan utilizando `StandardScaler` de `sklearn.preprocessing`.

   ```python
   from sklearn import preprocessing
   X = preprocessing.StandardScaler().fit(X).transform(X)
   ```

2. **División de Datos**:
   Los datos se dividen en conjuntos de entrenamiento y prueba usando `train_test_split` de `sklearn.model_selection`.

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
   ```

3. **Modelo SVM**:
   Se crea y entrena un clasificador de Máquina de Vectores de Soporte (SVM) usando `sklearn.svm` .

   ```python
   from sklearn import svm
   clf = svm.SVC(gamma=0.001, C=100.)
   clf.fit(X_train, y_train)
   ```

4. **Predicción**:
   El modelo se usa para predecir las etiquetas para los datos de prueba.

   ```python
   clf.predict(X_test)
   ```

5. **Matriz de Confusión**:
   Se imprime una matriz de confusión utilizando `confusion_matrix` de `sklearn.metrics`.

   ```python
   from sklearn.metrics import confusion_matrix
   print(confusion_matrix(y_test, yhat, labels=[1, 0]))
   ```

6. **Guardado del Modelo**:
   El modelo entrenado se guarda usando la librería `pickle`.

   ```python
   import pickle
   s = pickle.dumps(clf)
   ```