import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:/Training/Academy/Statistics (Python)/Cases/Bankruptcy/Bankruptcy.csv")

X = df.iloc[:,2:]
y = df.iloc[:,1]

########################## k FOLD CV ##############################

from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = StratifiedKFold(n_splits=5, random_state=2022,shuffle=True)

logreg = LogisticRegression()

results = cross_val_score(logreg, X, y, cv=kfold, 
                          scoring='roc_auc')
print(results.mean())


from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.05,random_state=2022)
clf.fit(X)
predictions = clf.predict(X)

########################################################
series_outliers = pd.Series(predictions,name="Outliers")
dt_outliers = pd.concat([df,series_outliers],axis=1)
only_outliers = dt_outliers[dt_outliers['Outliers']==-1]
wo_outliers = dt_outliers[dt_outliers['Outliers']!=-1]

X = wo_outliers.iloc[:,2:-1]
y = wo_outliers.iloc[:,1]

logreg = LogisticRegression()

results = cross_val_score(logreg, X, y, cv=kfold, 
                          scoring='roc_auc')
print(results.mean())
