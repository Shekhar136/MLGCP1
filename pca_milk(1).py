import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("milk.csv",index_col=0)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

## PCA Transform
pca = PCA()
principalcomponents =  pca.fit_transform(df_scaled)

## Variance of every component
print(pca.explained_variance_)

## Total variation
print(np.sum(pca.explained_variance_))

## Proportion of variation explained
print(pca.explained_variance_ratio_) 

## %age of variation explained
print(pca.explained_variance_ratio_*100)

## Cumulative sum
print(np.cumsum(pca.explained_variance_ratio_)*100)

pd_PC = pd.DataFrame(principalcomponents,
                     columns=['PC1','PC2','PC3','PC4','PC5'],
                     index=df.index)

plt.scatter(pd_PC['PC1'],pd_PC['PC2'])
plt.show()

############# USArrests ##############
from pca import pca
df = pd.read_csv("USArrests.csv", index_col=0)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled,columns=df.columns,index=df.index)

model = pca()

results = model.fit_transform(df_scaled)
fig, ax = model.biplot(label=True,legend=False)

########### Big Mart Sales ##################
import os
os.chdir(r"C:\Training\AV\Big Mart III")

train = pd.read_csv("train_v9rqX0R.csv")

avg_wt = train.groupby('Item_Type')['Item_Weight'].mean()
avg_vis = train.groupby('Item_Type')['Item_Visibility'].mean()
avg_mrp = train.groupby('Item_Type')['Item_MRP'].mean()
avg_sales = train.groupby('Item_Type')['Item_Outlet_Sales'].mean()

grp_data = pd.concat([avg_wt,avg_vis,avg_mrp,avg_sales], axis=1)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(grp_data)

## PCA Transform
pca = PCA()
principalcomponents =  pca.fit_transform(df_scaled)

## Cumulative sum
print(np.cumsum(pca.explained_variance_ratio_)*100)

model = pca()
df_scaled = pd.DataFrame(df_scaled,columns=grp_data.columns,index=grp_data.index)
results = model.fit_transform(df_scaled)
fig, ax = model.biplot(label=True,legend=False)

########## cars93 Stats by Mfg #########
from pca import pca
cars93 = pd.read_csv("cars93.csv")
small_cars = cars93[cars93['Type']=='Small']

avg_price = small_cars.groupby('Manufacturer')['Price'].mean()
avg_mpg_city = small_cars.groupby('Manufacturer')['MPG.city'].mean()
avg_mpg_high = small_cars.groupby('Manufacturer')['MPG.highway'].mean()
avg_mpg_es = small_cars.groupby('Manufacturer')['EngineSize'].mean()
avg_mpg_hp = small_cars.groupby('Manufacturer')['Horsepower'].mean()
avg_mpg_fuel = small_cars.groupby('Manufacturer')['Fuel.tank.capacity'].mean()
avg_mpg_rev = small_cars.groupby('Manufacturer')['Rev.per.mile'].mean()

grp_data = pd.concat([avg_price,avg_mpg_city,avg_mpg_high,avg_mpg_es,
                      avg_mpg_hp,avg_mpg_fuel,avg_mpg_rev], 
                     axis=1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(grp_data)

## PCA Transform
prin_comp = PCA()
principalcomponents =  prin_comp.fit_transform(df_scaled)

## Cumulative sum
print(np.cumsum(prin_comp.explained_variance_ratio_)*100)

model = pca()
df_scaled = pd.DataFrame(df_scaled,columns=grp_data.columns,index=grp_data.index)

results = model.fit_transform(df_scaled)
fig, ax = model.biplot(label=True,legend=False)
