import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
import os

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis")

kyphosis = pd.read_csv("Kyphosis.csv")
dum_kyp = pd.get_dummies(kyphosis, drop_first=True)
X = dum_kyp.drop('Kyphosis_present', axis=1)
y = dum_kyp['Kyphosis_present']
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
################### Over-Sampling(Naive) ###############
ros = RandomOverSampler(random_state=2022)
X_resampled, y_resampled = ros.fit_resample(X, y)

svm = SVC(probability=True, kernel='linear', random_state=2022)
params = {'C': np.linspace(0.001, 5, 10)}
gcv = GridSearchCV(svm, param_grid=params, scoring='roc_auc',
                          cv = kfold, verbose=3)
gcv.fit(X_resampled, y_resampled)
print(gcv.best_score_)
print(gcv.best_params_)

################# Over-Sampling(SMOTE) #################
smote = SMOTE(random_state=2022)
X_resampled, y_resampled = smote.fit_resample(X, y)

svm = SVC(probability=True, kernel='linear', random_state=2022)
params = {'C': np.linspace(0.001, 5, 10)}
gcv = GridSearchCV(svm, param_grid=params, scoring='roc_auc',
                          cv = kfold, verbose=3)
gcv.fit(X_resampled, y_resampled)
print(gcv.best_score_)
print(gcv.best_params_)

################# Over-Sampling(ADASYN) #################

adasyn = ADASYN(random_state=2022)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

svm = SVC(probability=True, kernel='linear', random_state=2022)
params = {'C': np.linspace(0.001, 5, 10)}
gcv = GridSearchCV(svm, param_grid=params, scoring='roc_auc',
                          cv = kfold, verbose=3)
gcv.fit(X_resampled, y_resampled)
print(gcv.best_score_)
print(gcv.best_params_)
