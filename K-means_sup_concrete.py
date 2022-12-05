import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold
import os

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")
df = pd.read_csv("Concrete_Data.csv")

X = df.drop(['Strength'],axis=1)
y = df['Strength']

# w/o Clustering
clf = xgb.XGBRegressor(random_state=2022)

kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
params = {'n_estimators':[50,100],
          'learning_rate':[0.001, 0.1, 0.4],
          'max_depth':[2, 4, 6]}
gcv = GridSearchCV(clf, param_grid=params,
                   cv = kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

# with Clustering
# Create scaler: scaler
scaler = StandardScaler()
Xscaled=scaler.fit_transform(X)

clustNos = [2,3,4,5,6,7,8,9,10]
silhouettes = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2022)
    model.fit(Xscaled)
    labels = model.predict(Xscaled)
    sil_score = silhouette_score(Xscaled,labels)
    print("k=", i, ", Sil =", sil_score)
    silhouettes.append(sil_score)
 
i_max = np.argmax(silhouettes)
best_k = clustNos[i_max]
print(best_k)

model = KMeans(n_clusters=best_k,random_state=2022)
model.fit(Xscaled)
labels = model.predict(Xscaled)
X['C'] = labels
X['C'] = X['C'].astype('category')

############ with one hot encoding ####################

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer,make_column_selector
ohc = OneHotEncoder()
ct = make_column_transformer((ohc,
       make_column_selector(dtype_include='category')),
                             ("passthrough",
                              make_column_selector(dtype_include=['int64','float64'])))
dum_np = ct.fit_transform(X)

print(ct.get_feature_names_out())
###########################################################


X_aug = pd.get_dummies(X)

gcv = GridSearchCV(clf, param_grid=params,
                   cv = kfold, scoring='r2')
gcv.fit(X_aug, y)
print(gcv.best_params_)
print(gcv.best_score_)




