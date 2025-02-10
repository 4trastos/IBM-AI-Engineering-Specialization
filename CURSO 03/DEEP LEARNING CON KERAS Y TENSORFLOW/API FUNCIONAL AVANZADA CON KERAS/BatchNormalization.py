from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.models import Model

# Define inputs
input_layer = Input(share=(20,))

# Add hidenn layer
hidden_layer = Dense(60, activation='relu')(input_layer)

# Add a BatchNormalization layer
batch_normalization = BatchNormalization()(hidden_layer)

# Add another hidden layer after BachtNormalization
hidden_layer2 = Dense(60, activation='relu')(batch_normalization)

# Add output layer
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Summary of the model
model.summary()
