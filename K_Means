
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('C:/Users/ADMIN/Downloads/iris.csv')
#print(dataset)
X=df.iloc[:,[3,4]].values
#print(X)

plt.scatter(df['PetalLengthCm'],df['PetalWidthCm'])
plt.show()
from sklearn.cluster import KMeans

wcss=[]
for i in range(1,11):
    kmeans= KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('elbow method')
plt.xlabel('k')
plt.ylabel('var')
plt.show()

kmeans= KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
Y_Kmeans=kmeans.fit_predict(X)
#print(Y_Kmeans)


plt.scatter(X[Y_Kmeans==0,0],X[Y_Kmeans==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[Y_Kmeans==1,0],X[Y_Kmeans==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[Y_Kmeans==2,0],X[Y_Kmeans==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroid')
plt.legend()
plt.show()
