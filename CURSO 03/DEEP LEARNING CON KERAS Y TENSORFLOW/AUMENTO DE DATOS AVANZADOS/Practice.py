#####################################################################
### Exercise 1: Apply and Visualize Different Augmentation Techniques

import numpy as np
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img 

import os
import urllib.request

# Crea la carpeta 'sample_images' si no existe
os.makedirs('sample_images', exist_ok=True)

# Lista de URLs de imágenes
image_urls = [
    'https://images.unsplash.com/photo-1551963831-b3b1ca40c98e',
    'https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e',
    'https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d'
]

# Descargar y guardar cada imagen
for i, url in enumerate(image_urls):
    image_path = f'sample_images/training_images{i+1}.jpg'
    urllib.request.urlretrieve(url, image_path)
    print(f'{image_path} descargada')


#####################################
#### Exercise 1

# Define the augmentation parameters  
datagen = ImageDataGenerator(
        rotation_range = 90,
        width_shift_range = 2,
        height_shift_range = 3,
        shear_range = 3,
        zoom_range = 6,
        horizontal_flip = True,
        fill_mode='nearest'
)

# Load and preprocess the dataset  
image_paths = [
    'sample_images/training_images1.jpg',  
    'sample_images/training_images2.jpg',  
    'sample_images/training_images3.jpg'
] 

# Train model
trainig_images = []
for image_path in image_paths:
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    trainig_images.append(img_array)
trainig_images = np.array(trainig_images)

# Generate and visualize augmented images 
i = 0
for batch in datagen.flow(trainig_images, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(array_to_img(batch[0]))
    plt.title(f'Augmented Image {i + 1}')  
    i += 1
    if i % 4 == 0:
        break

plt.show()

#####################################################################
### Exercise 2: Implement Feature-wise and Sample-wise Normalization

datagen = ImageDataGenerator(
    featurewise_center = True,
    featurewise_std_normalization = True,
    samplewise_center = True,
    samplewise_std_normalizaton = True
)

# Fit the ImageDataGenerator to the dataset
datagen.fit(trainig_images)

# Generar and visualize normalized images
i = 0
for batch in ImageDataGenerator(trainig_images, batch_size=32):
    plt.figure(i)
    imgplot = plt.imshow(array_to_img(batch[0]))
    plt.title(f'Normalization Options {i + 1}')
    i += 1
    if i % 4 == 0:
        break
plt.show()

#####################################################################
### Exercise 3: Create and Apply a Custom Data Augmentation Function

# Generar adds random noise
def add_random_noise(image):
    noise = np.random.normal(0,0,1, image.shape)
    return image + noise

# Define the augmentation parameters

datagen = ImageDataGenerator(preprocessing_funtions = add_random_noise)

# Generar and visualize normalized images
i = 0
for batch in datagen.flow(trainig_images, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(array_to_img(batch[0]))
    plt.title(f'Custom Options {i + 1}')
    i += 1
    if i % 4 == 0:
        break
plt.show()