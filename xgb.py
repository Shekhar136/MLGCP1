import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Boston Housing")
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt

boston = pd.read_csv("boston.csv")

X = boston.drop('medv', axis=1)
y = boston['medv']

gbm = xgb.XGBRegressor(random_state=2022)
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth': [2,3,4,5,6],
          'n_estimators':[50,100,150]}
gcv = GridSearchCV(gbm, param_grid=params, cv=kfold,
                   scoring='r2', verbose=3)

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

gbm = xgb.XGBClassifier(random_state=2022)
kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=2022)
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

################### Vehicle ##################################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Vehicle Silhouettes")
from sklearn.preprocessing import LabelEncoder
vehicle = pd.read_csv("Vehicle.csv")

X = vehicle.drop('Class', axis=1)
y = vehicle['Class']
le = LabelEncoder()
le_y = le.fit_transform(y)

gbm = xgb.XGBClassifier(random_state=2022,
                        objective='multi:softprob')
kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=2022)
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth': [2,3,4,5,6],
          'n_estimators':[50,100,150]}
gcv = GridSearchCV(gbm, param_grid=params, cv=kfold,
                   scoring='neg_log_loss', verbose=3)

gcv.fit(X, le_y)

pd_cv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_params_)
print(gcv.best_score_)

######## Random Forest
from sklearn.ensemble import RandomForestClassifier
kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=2022)
clf = RandomForestClassifier(random_state=2022)
params = {'max_features':[2,3,4,5,6]}
gcv = GridSearchCV(clf, param_grid=params,
                   scoring='neg_log_loss', cv=kfold, verbose=3)
gcv.fit(X,le_y)

print(gcv.best_params_)
print(gcv.best_score_)

#### XGBoost random forest classification
gbm = xgb.XGBRFClassifier(random_state=2022,
                        objective='multi:softprob')
kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=2022)
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth': [2,3,4,5,6],
          'n_estimators':[50,100,150]}
gcv = GridSearchCV(gbm, param_grid=params, cv=kfold,
                   scoring='neg_log_loss', verbose=3)

gcv.fit(X, le_y)

pd_cv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_params_)
print(gcv.best_score_)
