import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt

brupt = pd.read_csv("Bankruptcy.csv", index_col=0)

X = brupt.drop(['D','YR'], axis=1)
y = brupt['D']

gbm = GradientBoostingClassifier(random_state=2022)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth': [2,3,4,5,6],
          'n_estimators':[50,100,150]}
gcv = GridSearchCV(gbm, param_grid=params, cv=kfold,
                   scoring='roc_auc', verbose=3)

gcv.fit(X, y)

pd_cv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

imp=best_model.feature_importances_
i_sort = np.argsort(-imp)
sorted_imp = imp[i_sort]
sorted_col = X.columns[i_sort]

ind = np.arange(X.shape[1])
plt.bar(ind,sorted_imp)
plt.xticks(ind,(sorted_col),rotation=90)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()


############ Bank ############################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\bank")
bank = pd.read_csv("bank.csv", sep=";")
dum_bank = pd.get_dummies(bank, drop_first=True)

X = dum_bank.drop('y_yes', axis=1)
y = dum_bank['y_yes']

## Random Forest
kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=2022)
clf = RandomForestClassifier(random_state=2022)
params = {'max_features':[2,3,4,5,6]}
gcv = GridSearchCV(clf, param_grid=params,
                   scoring='roc_auc', cv=kfold, verbose=3)
gcv.fit(X,y)

print(gcv.best_params_)
print(gcv.best_score_)

#### GBM
gbm = GradientBoostingClassifier(random_state=2022)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth': [2,3,4,5,6],
          'n_estimators':[50,100,150]}
gcv = GridSearchCV(gbm, param_grid=params, cv=kfold,
                   scoring='roc_auc', verbose=3)

gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)
