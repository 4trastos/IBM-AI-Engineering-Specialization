import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Define inputs layer
input_layer = Input(shape=(20,))
print(input_layer)

#Add hidden Layers
hiden_layer1 = Dense(64, activation='relu')(input_layer)
hiden_layer2 = Dense(64, activation='relu')(hiden_layer1)

# output layer
output_layer = Dense(1, activation='sigmoid')(hiden_layer2)

# Create the model
model = Model(inputs = input_layer, outputs=output_layer)
model.summary()


# Comnpile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
import numpy as np
X_train = np.random.rand(1000, 20)
y_train = np.random.randint(2, size=(1000, 1))
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Test Data (evaluar modelo)
X_test = np.random.rand(200, 20)
y_test = np.random.randint(2, size=(200, 1))
loss, accuaracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuaracy}')

