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
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [1, 1]], cluster_std=0.9)

#### MOSTRAMOS EL DIAGRAMA DE DISPERSION ALEATORIO (scatter plot) #####
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.show()

### Una vez que tenemos los datos aleatorios configuramos los clusters K_means
# Inicializamos K_means

k_means = KMeans(init='k-means++', n_clusters=3, n_init=12)

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

fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means.cluster_centers_)), colors):
    my_members = (k_means_labels == k)
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.show()