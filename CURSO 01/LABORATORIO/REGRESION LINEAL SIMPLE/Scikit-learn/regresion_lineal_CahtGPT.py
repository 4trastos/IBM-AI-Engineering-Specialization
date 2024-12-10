import matplotlib.pyplot as plt     # para visualizar los gráficos
import pandas as pd                 # para leer datos de una tabla csv
import numpy as np                  # para realizar cálculos numéricos

# cargamos los datos
df = pd.read_csv("FuelConsumptionCo2.csv")
#print("Datos cargados correctamente\n")

# mostramos las primeras lineas del datasets
df.head()
#print(df.head())

# resumir los datos
df.describe()
#print(df.describe())

# Crear grafica relación entre el tamaño del motor y las emisiones de CO2
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()

# Seleccionamos las características (tamaño del motor) y la variable objetivo (emisiones de CO2)
x = df[["ENGINESIZE"]]          # Característica (predictor)
y = df["CO2EMISSIONS"]          # Variable objetivo (a predecir)

# Dividimos los datos en 80% entrenamiento y 20% prueba
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Creamos el modelo
lr_model = LinearRegression()

# Entrenamos el modelo con los datos de entrenamiento
lr_model.fit(x_train, y_train)

# Predecimos las emisiones de CO2 con los datos de prueba
predictions = lr_model.predict(x_test)

# Mostramos las predicciones y los valores reales
for i in range(len(y_test)):
    print(f"Real: {y_test.iloc[i]}, Prediction: {predictions[i]}")

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calcular el MAE, MSE y RMSE
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")