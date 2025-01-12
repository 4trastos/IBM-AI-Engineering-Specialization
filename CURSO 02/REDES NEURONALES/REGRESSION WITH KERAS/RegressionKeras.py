import pandas as pd
import numpy as np

import warnings
warnings.simplefilter('ignore', FutureWarning)

concrete_data = pd.read_csv("concrete_data.csv")
concrete_data.shape    # numero de columanas y lineas
concrete_data.head()   # muestra la base de datos

concrete_data.describe()       # Resumen estadístico de DataFrame
concrete_data.isnull().sum()   # Muestra si falan datos

concrete_data_columns = concrete_data.columns

###### SPLIT PARA TRAIN DEL MODELO #######

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

predictors.head()
target.head()

####### NORMALIZAR LOS DATOS #######

predictors_norm = (predictors - predictors.mean() / predictors.std())
predictors_norm.head()

n_cols = predictors_norm.shape[1]       # numero de predicciones

from keras import Sequential
from tensorflow.keras import Dense

def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# build the model
model = regression_model()
# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)