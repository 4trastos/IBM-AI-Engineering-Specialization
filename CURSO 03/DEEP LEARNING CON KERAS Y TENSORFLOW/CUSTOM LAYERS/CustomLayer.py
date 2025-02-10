import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential

# Define a custom layer
class CustomDenseLayer(Layer):
    def __init__(self, units=32):
        super(CustomDenseLayer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
    
    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
    
# Integrate the custom layer into a model
from tensorflow.keras.layers import Softmax

model = Sequential([
    CustomDenseLayer(128),
    CustomDenseLayer(10),
    Softmax()
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')
print("Model summary before building: ")
model.summary()

# Build the model to show parameters
model.build((1000, 20))
print("\nModel Sumary after building: ")
model.summary()

# Train the model
import numpy as np

# Generate random data 
X_train = np.random.random((1000, 20))
y_train = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate model

X_test = np.random.random((200, 20))
y_test = np.random.randint(10, size=(200, 1))

# Convert labels to categorical one-hot encoding
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10) 

# Evaluate model
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

#############################################
#### Exercise 1: Visualize Model Architecture

from tensorflow.keras.utils import plot_model

# Visualize the model arquitecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

#############################################
#### Exercise 2: Add Dropout Layer ##########

from tensorflow.keras.layers import Dropout

# Modify the model to include a Dropout Layer
model = Sequential([
    CustomDenseLayer(128),
    Dropout(0.5),
    CustomDenseLayer(10)
])

# Recompile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model again
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate model
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

###########################################################
#### Exercise 3: Adjust the Number of Units in Custom Layer

# Define a custom layer
class CustomDenseLayer(Layer):
    def __init__(self, units=128):
        super(CustomDenseLayer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
    
    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)

# Modify the number of units
model = Sequential([
    CustomDenseLayer(128),
    Dropout(0.5),
    CustomDenseLayer(10)
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=10, batch_size=32)
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')