#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import warnings
warnings.filterwarnings("ignore")


# In[8]:


#dataset = pd.read_csv('stretched blob clusters.csv')
dataset = pd.read_csv('ExifMetadata.csv')
X = dataset.iloc[:, [1, 2]].values


# In[9]:


X


# In[10]:


# cluster the data into five clusters
kmeans = KMeans (n_clusters=5)
kmeans.fit(X)
y_pred= kmeans.predict(X)
# plot the cluster assignments and cluster centers
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="plasma")
plt.scatter(kmeans.cluster_centers_[:, 0],
kmeans.cluster_centers_[:, 1],
marker='^',
c=[0, 1, 2, 3, 4],
s=100,
linewidth=2,
cmap="plasma")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# In[11]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled= scaler.fit_transform(X)
# cluster the data into five clusters
dbscan= DBSCAN(eps=0.123, min_samples= 2)
clusters = dbscan.fit_predict(X_scaled)
# plot the cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="plasma")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# In[6]:


#DBSCAN Clustering: (Optimum Knee)
# dotline located the best place

fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")
plt.savefig("knee.png", dpi=300)
print(distances[knee.knee])


# In[ ]:




