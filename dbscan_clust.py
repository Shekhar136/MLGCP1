from sklearn.cluster import DBSCAN
import numpy as np

import pandas as pd

milk = pd.read_csv("milk.csv",index_col=0)

from sklearn.preprocessing import StandardScaler
# Create scaler: scaler
scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

milkscaled = pd.DataFrame(milkscaled,
                          columns=milk.columns,
                          index=milk.index)


clust_DB = DBSCAN(eps=1, min_samples=4)
clust_DB.fit(milkscaled)
print(clust_DB.labels_)

clusterID = pd.DataFrame({'ClustID':clust_DB.labels_},index=milk.index)
clusteredData = pd.concat([milk,clusterID],
                          axis='columns')

clusteredData.groupby('ClustID').mean()
clusteredData.sort_values('ClustID')

from sklearn.metrics import silhouette_score
silhouette_score(milkscaled,clust_DB.labels_)

eps_range = [0.2,0.4,0.6,1]
mp_range = [2,3,4,5]
cnt = 0
a =[]
for i in eps_range:
    for j in mp_range:
        clust_DB = DBSCAN(eps=i, min_samples=j)
        clust_DB.fit(milkscaled)
        if len(set(clust_DB.labels_)) >= 2:
            cnt = cnt + 1
            sil_sc = silhouette_score(milkscaled,clust_DB.labels_)
            a.append([cnt,i,j,sil_sc])
            print(i,j,sil_sc)
    
a = np.array(a)
pa = pd.DataFrame(a,columns=['Sr','eps','min_pt','sil'])
print("Best Paramters:")
pa[pa['sil'] == pa['sil'].max()]

