import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
os.chdir(r"C:\Training\Kaggle\Competitions\Otto Product Classification")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import cross_val_score
train = pd.read_csv("train.csv", index_col=0)
print(train.shape)
print(train['target'].value_counts())
cts = train['target'].value_counts()
cts.plot(kind='bar')
plt.show()

X = train.drop('target', axis=1)
y = train['target']

lbl = LabelEncoder()
le_y = lbl.fit_transform(y)

### Fitting Logreg on entire train set
logreg = LogisticRegression(multi_class='ovr')
logreg.fit(X, le_y)

#### Applying on test set
test = pd.read_csv("test.csv", index_col=0)
y_pred_prob = logreg.predict_proba(test)

pd_pred_prob = pd.DataFrame(y_pred_prob,columns=list(lbl.classes_))

### Submission 
submit = pd.read_csv("sampleSubmission.csv")

submission = pd.concat([submit['id'],pd_pred_prob], axis=1)

### Fitting NB on entire train set
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X, le_y)

#### Applying on test set
test = pd.read_csv("test.csv", index_col=0)
y_pred_prob = nb.predict_proba(test)

pd_pred_prob = pd.DataFrame(y_pred_prob,columns=list(lbl.classes_))

### Submission 
submit = pd.read_csv("sampleSubmission.csv")

submission = pd.concat([submit['id'],pd_pred_prob], axis=1)
submission.to_csv("sbt_nb.csv", index=False)
