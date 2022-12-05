import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Wisconsin")
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

cancer = pd.read_csv("BreastCancer.csv", index_col=0)
dum_cancer = pd.get_dummies(cancer, drop_first=True)
X = dum_cancer.drop('Class_Malignant', axis=1)
y = dum_cancer['Class_Malignant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=2022)
svm = SVC(probability=True, kernel='linear')
svm.fit(X_train, y_train)

y_pred_prob = svm.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

######### Gird Search CV
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
svm = SVC(probability=True, kernel='linear')
params = {'C': np.linspace(0.001, 5, 10)}
gcv = GridSearchCV(svm, param_grid=params, scoring='roc_auc',
                          cv = kfold, verbose=3)
gcv.fit(X,y)
print(gcv.best_score_)
print(gcv.best_params_)
