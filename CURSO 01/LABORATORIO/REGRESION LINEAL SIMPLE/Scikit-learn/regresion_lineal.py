# importamos las librerías

import numpy as np                  # para realizar cálculos numéricos
import pandas as pd                 # para leer datos de una tabla csv
import matplotlib.pyplot as plt     # para visualizar los gráficos

# Leemos los datos
df = pd.read_csv("FuelConsumptionCo2.csv")

# Mostramos las primeras lineas de datos
df.head()

# Resumen estadístico de los datos
df.describe()

# Seleccionamos algunas características para usar en el modelo
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# Para representar gráficamente las características
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# Realizaremos las graficas de cada una de las características contra la emisión de CO2 para
# cuan lineal es su relación
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='green')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Tenemos que crear el split de datos de entrenamiento y test. 80% entrenamiento y 20% test
# Creamos una máscara paea seleccioanr fila aleatorias con np.random.rand()
msk = np.random.rand(len(df)) < 0.8     # booleana 0 1 / true false
train = cdf[msk]
test = cdf[~msk]

#### CREAMOS EL MODELO DE REGRESIÓN LINEAL SIMPLE #####

# Usamos la librería sklearn (scikit-learn) librería para realizar tareas de aprendizaje automático
from sklearn import linear_model
regr = linear_model.LinearRegression()          # regr lo usamos para train y realizar predicciones
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)                      # El método fit ajusta el modelo a los datos proporcionados. Ajusta la línea de regresión que mejor describe la relación entre los datos

# Printeamos los coeficientes
print (train_x, regr.coef_)                     # Almacena el coheficiente de regresión (cuanto cambia CO2)
print (train_y, regr.intercept_)                # Error potencial

# Podemos trazar la línea de AJUSTE sobre los datos
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#### EVSALUACIÓN DEL MODELO ####

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
predictions = regr.predict(test_x)

print ("Mean absolute error: %.2f" % np.mean(np.absolute(test_y - predictions)))
print ("Residual sum of squares (MSE): %.2f" % np.mean((test_y - predictions) ** 2))
print ("R2-score: %.2f" % r2_score(test_y, predictions))

# Imprimimos los coeficientes de la regresión
print(f"Coeficiente (Pendiente): {regr.coef_[0][0]}")  # Cuánto cambia CO2EMISSIONS por cada unidad de ENGINESIZE
print(f"Intersección (Intercepto): {regr.intercept_[0]}")  # Valor de CO2EMISSIONS cuando ENGINESIZE es 0


#### ¿Cómo puedes usar el modelo para predecir una variable dependiente en particular?
#### Supongamos que deseas predecir las emisiones de CO₂ para un tamaño de motor específico, por ejemplo, un ENGINESIZE de 3.5. Aquí te muestro cómo puedes hacerlo:
#### Crea un array con el valor que deseas predecir. El modelo espera un array de entrada, por lo que necesitas darle el valor en el formato adecuado.
#### Usa el método predict para generar la predicción.


#### Por ejemplo, si deseas predecir las emisiones para un tamaño de motor de 3.5:

engine_size = np.array([[3.5]])  # El modelo espera una entrada en forma de array bidimensional
predicted_emission = regr.predict(engine_size)

print(f"Predicción de emisiones de CO2 para un tamaño de motor de 3.5: {predicted_emission[0][0]}")
