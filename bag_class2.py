import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

brupt = pd.read_csv("Bankruptcy.csv", index_col=0)
X = brupt.drop(['D','YR'], axis=1)
y = brupt['D']

######### Grid Search CV : RBF
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
svm = SVC(probability=True, kernel='rbf',random_state=2022)
params = {'C': np.linspace(0.001, 5, 10),
          'gamma':np.linspace(0.001, 5, 10)}
gcv = GridSearchCV(svm, param_grid=params, scoring='roc_auc',
                          cv = kfold, verbose=3)
gcv.fit(X,y)
print(gcv.best_score_)
print(gcv.best_params_)

######### Grid Search CV : RBF with Bagging
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
svm = SVC(probability=True, kernel='rbf',random_state=2022)
bagging = BaggingClassifier(base_estimator=svm, random_state=2022)
params = {'base_estimator__C': np.linspace(0.001, 5, 10),
          'base_estimator__gamma':np.linspace(0.001, 5, 10),
          'n_estimators':[10, 50, 100]}
gcv = GridSearchCV(bagging, param_grid=params, scoring='roc_auc',
                          cv = kfold, verbose=3)
gcv.fit(X,y)
print(gcv.best_score_)
print(gcv.best_params_)

