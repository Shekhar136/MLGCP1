import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import tree
import matplotlib.pyplot as plt

############ HR Data #############
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")
hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=2022)
clf = RandomForestClassifier(random_state=2022)
params = {'max_features':[2,3,4,5,6]}
gcv = GridSearchCV(clf, param_grid=params,
                   scoring='roc_auc', cv=kfold, verbose=3)
gcv.fit(X,y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

print(best_model.feature_importances_)
ind = np.arange(18)
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns),rotation=90)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()