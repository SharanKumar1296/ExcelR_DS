#===================================================================================
""" Using the different ways of Cluster Analysis on the crime dataset and drawing 
inferences. """

import pandas as pd 

df = pd.read_excel("EastWestAirlines.xlsx",sheet_name="data")

df.head()
df.shape
df.dtypes

from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

df.iloc[:,1:] = mm.fit_transform(df.iloc[:,1:])

X = df.iloc[:,1:]

#Using K-Means Clustering
#Building an elbow plot
from sklearn.cluster import KMeans
clust = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(X)
    clust.append(kmeans.inertia_)

import matplotlib.pyplot as plt 
plt.scatter(x=range(1, 11), y=clust,color='red')
plt.plot(range(1, 11), clust,color='black')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()
#We can see using the elbow curve that the best number of clusters is 4.

kmeans = KMeans(n_clusters=5,n_init=20)

# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
Y = kmeans.predict(X)
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts() #3    1032
                        #0     868
                        #1     808
                        #4     673
                        #2     618

#We can observe that using Kmeans clustering we are able to form clusters in 
#accordance to the elbow plot information 

#Using Agglomerative Clustering
import scipy.cluster.hierarchy as shc 
plt.figure(figsize=(10,7))
plt.title("Dendograms")
dend = shc.dendrogram(shc.linkage(X,method="single"))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="single")
Y = cluster.fit_predict(X)

Y_new = pd.DataFrame(Y)
Y_new.value_counts()  # 0    2518
                      # 1    1478
                      # 2       1
                      # 3       1
                      # 4       1

plt.figure(figsize=(10,7))
plt.title("Dendograms")
dend = shc.dendrogram(shc.linkage(X,method="complete"))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="complete")
Y = cluster.fit_predict(X)

Y_new = pd.DataFrame(Y)
Y_new.value_counts() # 0    2495
                     # 2    1144
                     # 1     325
                     # 4      31
                     # 3       4

plt.figure(figsize=(10,7))
plt.title("Dendograms")
dend = shc.dendrogram(shc.linkage(X,method="average"))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="average")
Y = cluster.fit_predict(X)

Y_new = pd.DataFrame(Y)
Y_new.value_counts() # 1    2518
                     # 0    1468
                     # 4       8
                     # 3       4
                     # 2       1

#Above we have used the multiple methods of agglomerative clustering and have come up
#with 4 clusters for each.

#Using DBSCAN to create clusters
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=1.25,min_samples=3)  

db.fit(X)

Y = db.labels_ #The labels created after the clustering 
pd.DataFrame(Y).value_counts() 

db = DBSCAN(eps=0.5,min_samples=2)  

db.fit(X)

Y = db.labels_ #The labels created after the clustering 
pd.DataFrame(Y).value_counts() 

#Tried and tested multiple epsilon values and also multiple samples. We can see
#the formation of clusters in the second split.

df["cluster"] = pd.DataFrame(Y)

noise_points = df[df["cluster"]==-1] #Separating the noise points from the data 

final_data = df[df["cluster"]!=-1] #Creating a new dataframe without outliers

#We have tried and tested multiple clustering techniques and come up with clusters for 
#the given dataset.

#===================================================================================