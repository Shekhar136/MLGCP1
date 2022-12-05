import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Wisconsin")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

cancer = pd.read_csv("BreastCancer.csv", index_col=0)
dum_cancer = pd.get_dummies(cancer, drop_first=True)
X = dum_cancer.drop('Class_Malignant', axis=1)
y = dum_cancer['Class_Malignant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=2022)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred_prob = logreg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

######### K-Folds CV
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
logreg = LogisticRegression()
results = cross_val_score(logreg, X, y, scoring='roc_auc',
                          cv = kfold)
print(results.mean())

### HR Data
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)

X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

######### K-Folds CV
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
logreg = LogisticRegression(max_iter=1000)
results = cross_val_score(logreg, X, y, scoring='roc_auc',
                          cv = kfold)
print(results.mean())

#### Grid Search CV
params = {'penalty':['l1','l2','elasticnet','none'],
          'solver':['newton-cg','lbfgs','liblinear',
                    'sag','saga']}
gcv = GridSearchCV(logreg,param_grid=params,
                   scoring='roc_auc',cv=kfold, verbose=3)
gcv.fit(X, y)

pd_cv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_params_)
print(gcv.best_score_)

################## Image Segmentation #########################

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Image Segmentation")
from sklearn.preprocessing import LabelEncoder

imag_seg = pd.read_csv("Image_Segmention.csv")

X = imag_seg.drop('Class', axis=1)
y = imag_seg['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)

#### Grid Search CV
params = {'penalty':['l1','l2','elasticnet','none'],
          'multi_class':['ovr','multinomial']}
logreg = LogisticRegression(max_iter=1000)
gcv = GridSearchCV(logreg,param_grid=params,
                   scoring='neg_log_loss',cv=kfold, verbose=3)
gcv.fit(X, le_y)

pd_cv = pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)

### roc 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=le_y, random_state=2022)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred_prob = logreg.predict_proba(X_test)
# OVO
print(roc_auc_score(y_test, y_pred_prob, multi_class='ovo'))
# OVR
print(roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))

### OVO
gcv = GridSearchCV(logreg,param_grid=params,
                   scoring='roc_auc_ovo',cv=kfold, verbose=3)
gcv.fit(X, le_y)

pd_cv = pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)

### OVR
gcv = GridSearchCV(logreg,param_grid=params,
                   scoring='roc_auc_ovr',cv=kfold, verbose=3)
gcv.fit(X, le_y)

pd_cv = pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)

