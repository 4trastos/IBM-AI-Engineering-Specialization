import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model

# Set environment variables to suppress tersorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Use loggin to supress Tensroflow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Define input layer
input_layer = Input(shape=(28, 28, 1))

# Add convolutional and transpose convolutional layers
conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
transpose_conv_layer = Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(conv_layer)

# Create the model
model = Model(inputs = input_layer, outputs = transpose_conv_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model wiht Generate synthetic trainig data
X_train = np.random.rand(1000, 28, 28, 1)
y_train = X_train # Para la reconstrucción, el objetivo es la entrada.

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
# Generate synthetic test data
X_test = np.random.rand(200, 28, 28, 1)
y_test = X_test

loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# Visualize the Results
# Predict on test data
y_pred = model.predict(X_test)

# Plot de sample imgaes
n = 10 # Number of the shamples display

plt.figure(figsize=(20, 4))

for i in range(n): 

    # Display original 
    ax = plt.subplot(2, n, i + 1) 
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original") 
    plt.axis('off') 
    # Display reconstruction 
    ax = plt.subplot(2, n, i + 1 + n) 
    plt.imshow(y_pred[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show() 


###############################################################
###########  Exercise 1: Experiment with Different Kernel Sizes

conv_layer2 = Conv2D(filters=32, kernel_size=(6,6), activation='relu', padding='same')(input_layer)
transpose_conv_layer2 = Conv2DTranspose(filters=1, kernel_size=(6, 6), activation='sigmoid', padding='same')(conv_layer2)

model = Model(inputs = input_layer, outputs=transpose_conv_layer2)

model.compile(optimizer='adam', loss='mean_squared_error')

X_train = np.random.rand(1000, 28, 28, 1)
y_train = X_train
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

X_test = np.random.rand(200, 28, 28, 1)
y_test = X_test
loss2 = model.evaluate(X_test, y_test)

print(f'Test loss 2: {loss2}')

###############################################################
###########  Exercise 2: Add Dropout Layers ###################

conv_layer2 = Conv2D(filters=32, kernel_size=(6,6), activation='relu', padding='same')(input_layer)
dropout_layer = Dropout(0.5)(conv_layer2)
transpose_conv_layer2 = Conv2DTranspose(filters=1, kernel_size=(6, 6), activation='sigmoid', padding='same')(dropout_layer)

model = Model(inputs = input_layer, outputs = transpose_conv_layer2)

model.compile(optimizer='adam', loss='mean_squared_error')

X_train = np.random.rand(1000, 28, 28, 1)
y_train = X_train
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

X_test = np.random.rand(200, 28, 28, 1)
y_test = X_test
loss3 = model.evaluate(X_test, y_test)

print(f'Test loss 3: {loss}')
print(f'Test loss 3: {loss2}')
print(f'Test loss 3: {loss3}')

###############################################################
###########  Exercise 3: Use Different Activation Functions  ##

conv_layer = Conv2D(filters=32, kernel_size=(6, 6), activation='tanh', padding='same')(input_layer)
dropout_layer = Dropout(0.5)(conv_layer)
transpose_conv_layer2 = Conv2DTranspose(filters=1, kernel_size=(6, 6), activation='tanh', padding='same')(dropout_layer)

model = Model(inputs = input_layer, outputs = transpose_conv_layer2)

model.compile(optimizer='adam', loss='mean_squared_error')

X_train = np.random.rand(1000, 28, 28, 1)
y_train = X_train
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

X_test = np.random.rand(200, 28, 28, 1)
y_test = X_test
loss4 = model.evaluate(X_test, y_test)

print(f'Test loss 3: {loss}')
print(f'Test loss 3: {loss2}')
print(f'Test loss 3: {loss3}')
print(f'Test Loss 4: {loss4}')