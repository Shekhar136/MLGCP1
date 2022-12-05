import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("milk.csv",index_col=0)
########################################################
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

wss = []
for i in np.arange(2,11):
    model = KMeans(n_clusters=i,random_state=2022)
    model.fit(df_scaled)
    wss.append(model.inertia_)

plt.plot(np.arange(2,11),wss)
plt.title("Scree Plot")
plt.xlabel("No. of Clusters")
plt.ylabel("WSS")
plt.show()

best_k = 4
model = KMeans(n_clusters=best_k,random_state=2022)
model.fit(df_scaled)
labels = model.predict(df_scaled)

df["Cluster"] = labels
df.sort_values(by="Cluster")

# Centroids of Original Data:
df.groupby("Cluster").mean()

