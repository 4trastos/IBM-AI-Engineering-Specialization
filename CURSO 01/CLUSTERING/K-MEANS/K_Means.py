# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

np.random.seed()

### CREAMOS LOS GRUPOS ALEATORIOS CON make_blobs ####
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

#### MOSTRAMOS EL DIAGRAMA DE DISPERSION ALEATORIO (scatter plot) #####
plt.scatter(X[:, 0], X[:, 1], marker='.')
#plt.show()

### Una vez que tenemos los datos aleatorios configuramos los clusters K_means
# Inicializamos K_means

k_means = KMeans(init='k-means++', n_clusters=4, n_init=12)

# Ajustamos el modelo con la Matriz X

k_means.fit(X)

# Etiquetamos cada punto

k_means_labels = k_means.labels_
k_means_labels

# Obtenemos las coordenadas

k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

#####################################################
####### Creamos la gráfica de nuestro modelo ########
#####################################################

# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())
#plt.show()

#### PROCEDEMOS A SEGMENTAR LOS CLIENTES DE UNA DATA BASE ####

cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()
print(cust_df.head())

#### PRE-PROCESSING DATA BASE ####
# Busca si hay algun alguna variable categórica. En este caso Address

df = cust_df.drop('Address', axis=1)
df

# Normalizamos el conjunto de datos
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]         # Selecciona todas las filas y omite la primera columna si es un ID o algo no numérico
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

# Modelamos

clusterNum = 3
k_means = KMeans(init='k-means++', n_clusters = clusterNum, n_init=12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

df['Clus_km'] = labels
df.head(5)
print(df.head(5))

df.groupby('Clus_km').mean()

area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

# Gráfico 3D

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education', fontsize=12)
ax.set_ylabel('Age', fontsize=12)
ax.set_zlabel('Income', fontsize=12)

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(float))
plt.show()