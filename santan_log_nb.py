import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
os.chdir(r"C:\Training\Kaggle\Competitions\Santander Customer Satisfaction")

from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import cross_val_score
train = pd.read_csv("train.csv", index_col=0)
print(train.shape)
print(train['TARGET'].value_counts())
cts = train['TARGET'].value_counts()
cts.plot(kind='bar')
plt.show()

X = train.drop('TARGET', axis=1)
y = train['TARGET']

## K-Fold cv with log reg
logreg = LogisticRegression()
kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=2022)
results = cross_val_score(logreg, X, y, 
                          scoring='roc_auc', cv=kfold)
print(results.mean())

### Fitting Logreg on entire train set
logreg = LogisticRegression()
logreg.fit(X, y)

#### Applying on test set
test = pd.read_csv("test.csv", index_col=0)
y_pred_prob = logreg.predict_proba(test)[:,1]

### Submission 
submit = pd.read_csv("sample_submission.csv")
submit['TARGET'] = y_pred_prob

submit.to_csv("sbt_log.csv", index=False)

### Fitting GaussianNB on entire train set
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X, y)

#### Applying on test set
test = pd.read_csv("test.csv", index_col=0)
y_pred_prob = nb.predict_proba(test)[:,1]

### Submission 
submit = pd.read_csv("sample_submission.csv")
submit['TARGET'] = y_pred_prob

submit.to_csv("sbt_nb.csv", index=False)
