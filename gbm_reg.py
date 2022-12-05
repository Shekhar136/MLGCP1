import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Boston Housing")
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

boston = pd.read_csv("boston.csv")

X = boston.drop('medv', axis=1)
y = boston['medv']

gbm = GradientBoostingRegressor(random_state=2022)
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

########################### Concrete #############################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")

concrete = pd.read_csv("Concrete_Data.csv")

X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

gbm = GradientBoostingRegressor(random_state=2022)
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