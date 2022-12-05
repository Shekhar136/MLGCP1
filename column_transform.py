import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold,  train_test_split 
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer,make_column_selector


df = pd.read_csv("C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy\Bankruptcy.csv")
dum_df = pd.get_dummies(df, drop_first=True)

X = df.drop(['D','NO','YR'],axis=1)
y = df['D']


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2022,
                                                    stratify=y)
# with Clustering
# Create scaler: scaler
scaler = StandardScaler()
trnscaled=scaler.fit_transform(X_train)

clustNos = [2,3,4,5,6,7,8,9,10]
silhouettes = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2022)
    model.fit(trnscaled)
    labels = model.predict(trnscaled)
    sil_score = silhouette_score(trnscaled,labels)
    print("k=", i, ", Sil =", sil_score)
    silhouettes.append(sil_score)
 
i_max = np.argmax(silhouettes)
best_k = clustNos[i_max]
print(best_k)

model = KMeans(n_clusters=best_k,random_state=2022)
model.fit(trnscaled)
labels = model.predict(trnscaled)
X_train['C'] = labels
X_train['C'] = X_train['C'].astype('category')
ohc = OneHotEncoder()
ct = make_column_transformer((ohc,
       make_column_selector(dtype_include='category')),
                             ("passthrough",
                              make_column_selector(dtype_include=['int64','float64'])))
dum_np = ct.fit_transform(X_train)

print(ct.get_feature_names_out())

dum_trn_pd = pd.DataFrame(dum_np,
                          columns=ct.get_feature_names_out())


tstscaled = scaler.transform(X_test)
labels = model.predict(tstscaled)
X_test['C'] = labels
X_test['C'] = X_test['C'].astype('category')

dum_tst = ct.transform(X_test)

dum_tst_pd = pd.DataFrame(dum_tst,
                          columns=ct.get_feature_names_out())


