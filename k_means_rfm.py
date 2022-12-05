import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Recency Frequency Monetary")
rfm = pd.read_csv("rfm_data_customer.csv", index_col=0)
rfm.drop('most_recent_visit',axis=1, inplace=True)

# Create scaler: scaler
scaler = StandardScaler()
rfmscaled=scaler.fit_transform(rfm)


clustNos = [2,3,4,5,6,7,8,9,10]
silhouettes = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2022)
    model.fit(rfmscaled)
    labels = model.predict(rfmscaled)
    sil_score = silhouette_score(rfmscaled,labels)
    print("k=", i, ", Sil =", sil_score)
    silhouettes.append(sil_score)
 
i_max = np.argmax(silhouettes)
best_k = clustNos[i_max]
print(best_k)

# Import pyplot
import matplotlib.pyplot as plt

plt.plot(clustNos, silhouettes, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Silhouette Score')
plt.xticks(clustNos)
plt.show()


# Create a KMeans instance with clusters: model
model = KMeans(n_clusters=best_k,random_state=2022)

# Fit model to points
model.fit(rfmscaled)
# Cluster Centroids
print(model.cluster_centers_)
#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(rfmscaled)

rfm['ClusterID'] = labels

rfm.sort_values(by='ClusterID', inplace=True)

rfm.groupby('ClusterID').mean()


