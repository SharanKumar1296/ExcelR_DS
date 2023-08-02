#==============================================================================
""" Performing Principal Component Analysis on the wine dataset and perform 
clustering using first 3 principal component scores (both heirarchial and k mean clustering)
and obtain optimum number of clusters and check whether we have obtained same number 
of clusters with the original data."""

import pandas as pd
df = pd.read_csv("wine.csv")

list(df)

#standardizing the X variables
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

df.iloc[:,1:] = mm.fit_transform(df.iloc[:,1:])

X = df.iloc[:,1:]

pd.set_option("display.float_format",lambda x:"%.5f"%x)

#Performing Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA()

pc = pca.fit_transform(X)

pd.DataFrame(pc)

df1 = pd.DataFrame(pca.explained_variance_ratio_)

X = pc[:,0:3]

#Using Agglomerative Clustering
#Building a Dendogram for visualization
import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as shc 
plt.figure(figsize=(10,7))
plt.title("Dendograms")
dend = shc.dendrogram(shc.linkage(X,method="single"))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="complete")
Y = cluster.fit_predict(X)

Y_new = pd.DataFrame(Y)
Y_new.value_counts()

#We can see that using this clustering method we are unable to form clusters in 
#accordance to the proportions of that of the original dataset

#Using K-Means Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,n_init=20)

# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
Y = kmeans.predict(X)
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

#We can observe that using Kmeans clustering we are able to form clusters in a 
#similar proportion to that of the original dataset 

#Building the elbow plot 
clust = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(X)
    clust.append(kmeans.inertia_)


plt.scatter(x=range(1, 11), y=clust,color='red')
plt.plot(range(1, 11), clust,color='black')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()
#We can see using the elbow curve that the best number of clusters is 3 in accordance
#with the original data

#==============================================================================