import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")
df = pd.read_csv("Bankruptcy.csv", index_col=0)
X = df.drop('D', axis=1)
y = df.D
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,
                                                    random_state=2022)
svm = SVC(probability=True, random_state=2022)
scl_trn = scaler.fit_transform(X_train)
pca = PCA(n_components=0.9, random_state=2022)
prn_trn =  pca.fit_transform(scl_trn)
## Cumulative sum
print(np.cumsum(pca.explained_variance_ratio_)*100)
#prn_trn = prn_trn[:,:7]
svm.fit(prn_trn,y_train)

##### test set
scl_tst = scaler.transform(X_test)
prn_trn =  pca.transform(scl_tst)
#prn_trn = prn_trn[:,:7]

y_pred_prob = svm.predict_proba(prn_trn)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

##### Using Pipeling
from sklearn.pipeline import Pipeline
scaler = StandardScaler()
pca = PCA(n_components=0.9, random_state=2022)
svm = SVC(probability=True, random_state=2022)

pipe = Pipeline([('scl', scaler),('PC', pca),('SVM', svm)])

pipe.fit(X_train, y_train)
y_pred_prob = pipe.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

#### Grid Search CV
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=2022)
scaler = StandardScaler()
pca = PCA(random_state=2022)
svm = SVC(probability=True, random_state=2022)
pipe = Pipeline([('scl', scaler),('PC', pca),('SVM', svm)])
params = {'PC__n_components':[0.75, 0.8, 0.85, 0.9, 0.95],
          'SVM__C':[0.5, 1, 1.5],
          'SVM__gamma':['scale','auto']}

gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   scoring='roc_auc', cv=kfold)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

########## HR Data #################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")

df = pd.read_csv("HR_comma_sep.csv")
dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.drop('left', axis=1)
y = dum_df.left

scaler = StandardScaler()
pca = PCA(random_state=2022)
svm = SVC(probability=True, random_state=2022)

pipe = Pipeline([('scl', scaler),('PC', pca)])

prin_comps = pipe.fit(X)

plt.plot(np.arange(1, X.shape[1]+1), np.cumsum(pca.explained_variance_ratio_)*100)
plt.xlabel("PCs")
plt.ylabel("% age Variation Explained")
plt.show()

#### Grid Search CV
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=2022)
scaler = StandardScaler()
pca = PCA(random_state=2022)
svm = SVC(probability=True, random_state=2022)
pipe = Pipeline([('scl', scaler),('PC', pca),('SVM', svm)])
params = {'PC__n_components':[0.75, 0.8, 0.85, 0.9, 0.95],
          'SVM__C':[0.5, 1, 1.5],
          'SVM__gamma':['scale','auto']}

gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   scoring='roc_auc', cv=kfold)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

############ Glass #####################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Glass Identification")

df = pd.read_csv("Glass.csv")
X = df.drop('Type', axis=1)
y = df.Type

scaler = StandardScaler()
pca = PCA(random_state=2022)

pipe = Pipeline([('scl', scaler),('PC', pca)])

prin_comps = pipe.fit_transform(X)

pd_prin_comps = pd.DataFrame(prin_comps[:,:2],
                             columns=['PC1','PC2'])
pd_prin_comps['Type'] = y.astype('category')

sns.scatterplot(x='PC1', y='PC2',hue='Type',
                data=pd_prin_comps)
plt.show()

