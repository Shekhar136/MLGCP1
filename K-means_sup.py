import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

df = pd.read_csv("C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy\Bankruptcy.csv")
dum_df = pd.get_dummies(df, drop_first=True)

X = df.drop(['D','NO','YR'],axis=1)
y = df['D']

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2022,
                                                    stratify=y)

# w/o Clustering
rf = RandomForestClassifier(random_state=2022)
rf.fit(X_train, y_train)

y_pred_prob = rf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

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
X_trn_aug = pd.get_dummies(X_train)

tstscaled = scaler.transform(X_test)
labels = model.predict(tstscaled)
X_test['C'] = labels
X_test['C'] = X_test['C'].astype('category')
X_tst_aug = pd.get_dummies(X_test)

rf = RandomForestClassifier(random_state=2022)
X_trn_aug.drop('C_3', axis=1, inplace=True)
rf.fit(X_trn_aug, y_train)

y_pred_prob = rf.predict_proba(X_tst_aug)[:,1]
print(roc_auc_score(y_test, y_pred_prob))




