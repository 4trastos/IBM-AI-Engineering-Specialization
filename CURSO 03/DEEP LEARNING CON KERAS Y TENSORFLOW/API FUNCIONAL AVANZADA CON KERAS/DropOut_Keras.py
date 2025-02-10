from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model

# Define input layer
input_layer = Input(shape=(20,))

# Add a hidden layer
hidden_layer = Dense(64, activate='relu')(input_layer)

# Add Dropout layer
Dropout_layer = Dropout(rate='0.5')(hidden_layer)

# Add another hidden layer after dropout
hidden_layer2 = Dense(64, activation='relu')(Dropout_layer)

# Define the oputput layer
Output_layer = Dense(1, activation='sigmoid')(hidden_layer2)

# Create the model
model = Model(inputs=input_layer, outputs=Output_layer)

# Summary of the model
model.summary()
