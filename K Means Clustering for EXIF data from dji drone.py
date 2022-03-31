#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# In[2]:


#reading exif data set
meta = pd.read_csv('ExifMetadata.csv')


# In[3]:


meta


# In[4]:


meta.info()


# In[5]:


meta1 = meta.iloc[:, [1,2,5,6,7,8,9,10,11]]
correlation = meta1.corr() # find correlation between the features

sns.heatmap(correlation, square = True) #seaborn visualization package


# In[6]:


#find relation between ErrorDistance and Gimball Pitch degree
ss = StandardScaler()
X = meta.iloc[:, [2,3]].values #ErrorDistance vs Gimball Pitch degree / Latitude vs Longitude
XX = ss.fit_transform(X)
XX


# In[7]:


from sklearn.cluster import KMeans
wcss = []
for i in range (1, 11):
    kmeans = KMeans (n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(XX)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[8]:


#Choose 3

kmeans = KMeans (n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(XX)
y_kmeans


# In[12]:


# to visualize the K-means clustering
sns.scatterplot(XX[:,0], XX[:,-1], hue=['cluster-{}'.format(x) for x in y_kmeans])
#plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 60, c = 'red', label = 'Cluster1')
#plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 60, c = 'blue', label = 'Cluster2')
#plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 60, c = 'green', label = 'Cluster3')
#plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 60, c = 'violet', label = 'Cluster4')
#plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 200, c ='skyblue', label ='Centroids')
plt.title ('Clusters and Centroids')
plt.xlabel ('Latitude')
plt.ylabel ('Longitude')
plt.legend()
plt.show()


# In[ ]:




