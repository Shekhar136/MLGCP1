# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

os.chdir(r"C:\Training\AV\Big Mart III")

train = pd.read_csv("train_v9rqX0R.csv")

avg_wt = train.groupby('Item_Type')['Item_Weight'].mean()
avg_vis = train.groupby('Item_Type')['Item_Visibility'].mean()
avg_mrp = train.groupby('Item_Type')['Item_MRP'].mean()
avg_sales = train.groupby('Item_Type')['Item_Outlet_Sales'].mean()

grp_data = pd.concat([avg_wt,avg_vis,avg_mrp,avg_sales], axis=1)

scaler = StandardScaler()
grp_data_scaled=scaler.fit_transform(grp_data)

# Calculate the linkage: mergings
mergings = linkage(grp_data_scaled,method='average')

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
dendrogram(mergings,
           labels=np.array(grp_data.index),
           leaf_rotation=90,
           leaf_font_size=15,
)

plt.show()