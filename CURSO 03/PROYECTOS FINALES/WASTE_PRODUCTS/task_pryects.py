


# Task 1: Print the version of tensorflow
import tensorflow as tf
print(tf.__version__)



# Task 2: Create a `test_generator` using the `test_datagen` object
test_generator = test_datagen.flow_from_directory(
    directory = path_test,
    class_mode = 'binary',
    seed = seed,
    batch_size = batch_size,
    shuffle = False,
    target_size = (img_rows, img_cols)
)



# Task 3: print the length of the `train_generator`
print(f"Lenght of Train Generator: {len(train_generator)}")



# Task 4: print the summary of the model
model.summary()




for layer in basemodel.layers: 
    layer.trainable = False

# Task 5: Compile the model
model.compile(
    loss = 'binary_crossentropy',
    optimizer = optimizers.RMSprop(learning_rate=1e-4),
    metrics = ['accuracy']
)



import matplotlib.pyplot as plt
history = extract_feat_model

# Task 6: Plot accuracy curves for training and validation sets
plt.figure(figsize=(5, 5))
plt.plot(history.history['accuracy'], label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



history = fine_tune_model

# Task 7: Plot loss curves for training and validation sets (fine tune model)
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


history = fine_tune_model

# Task 8: Plot accuracy curves for training and validation sets  (fine tune model)
plt.figure(figsize=(5, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



# Task 9: Plot a test image using Extract Features Model (index_to_plot = 1)
index_to_plot = 1
plot_image_with_title(
    image = test_imgs[index_to_plot].astype('uint8'),
    model_name = 'Extract Features Model',
    actual_label=test_labels[index_to_plot],
    predicted_label=predictions_extract_feat_model[index_to_plot]
)


# Task 10: Plot a test image using Fine-Tuned Model (index_to_plot = 1)
index_to_plot = 1
plot_image_with_title(
    image = test_imgs[index_to_plot].astype('uint8'),
    model_name = 'Fine-Tuned Model',
    actual_label=test_labels[index_to_plot],
    predicted_label=predictions_fine_tune_model[index_to_plot]
)