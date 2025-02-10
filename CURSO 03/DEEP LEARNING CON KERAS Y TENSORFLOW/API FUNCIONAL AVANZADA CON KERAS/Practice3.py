from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np

input_layer = Input(shape=(20,))

hidden_layer = Dense(60, activation='relu')(input_layer)
batch_01 = BatchNormalization()(hidden_layer)
hidden_layer2 = Dense(60, activation='relu')(batch_01)
batch_02 = BatchNormalization()(hidden_layer2)

output_layer = Dense(1, activation='sigmoid')(batch_02)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train = np.random.rand(1000, 20)
y_train = np.random.randint(2, size=(1000, 1))
model.fit(X_train, y_train, epochs=10, batch_size=32)

X_test = np.random.rand(200, 20)
y_test = np.random.randint(2, size=(200, 1))
loss, accuaracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuaracy}')