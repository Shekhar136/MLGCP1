import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Wisconsin")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

cancer = pd.read_csv("BreastCancer.csv", index_col=0)
dum_cancer = pd.get_dummies(cancer, drop_first=True)
X = dum_cancer.drop('Class_Malignant', axis=1)
y = dum_cancer['Class_Malignant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=2022)
da = LinearDiscriminantAnalysis()
da.fit(X_train, y_train)

y_pred_prob = da.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

######### K-Folds CV Linear
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
da = LinearDiscriminantAnalysis()
results = cross_val_score(da, X, y, scoring='roc_auc',
                          cv = kfold)
print(results.mean())

######### K-Folds CV Quadratic
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
da = QuadraticDiscriminantAnalysis()
results = cross_val_score(da, X, y, scoring='roc_auc',
                          cv = kfold)
print(results.mean())

######### Santander
os.chdir(r"C:\Training\Kaggle\Competitions\Santander Customer Satisfaction")
train = pd.read_csv("train.csv", index_col=0)

X = train.drop('TARGET', axis=1)
y = train['TARGET']

da = LinearDiscriminantAnalysis()
da.fit(X,y)
test = pd.read_csv("test.csv", index_col=0)
y_pred_prob = da.predict_proba(test)[:,1]

### Submission 
submit = pd.read_csv("sample_submission.csv")
submit['TARGET'] = y_pred_prob

submit.to_csv("sbt_lda.csv", index=False)


########## Otto
from sklearn.preprocessing import LabelEncoder
os.chdir(r"C:\Training\Kaggle\Competitions\Otto Product Classification")

train = pd.read_csv("train.csv", index_col=0)
X = train.drop('target', axis=1)
y = train['target']

lbl = LabelEncoder()
le_y = lbl.fit_transform(y)
da = QuadraticDiscriminantAnalysis()
da.fit(X,le_y)
test = pd.read_csv("test.csv", index_col=0)
y_pred_prob = da.predict_proba(test)
pd_pred_prob = pd.DataFrame(y_pred_prob,
                            columns=list(lbl.classes_))

### Submission 
submit = pd.read_csv("sampleSubmission.csv")

submission = pd.concat([submit['id'],pd_pred_prob], axis=1)
submission.to_csv("submit_qda.csv", index=False)
