import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

################## Concrete Strength #####################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")

concrete = pd.read_csv("Concrete_Data.csv")

X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

######### Grid Search CV : Tree
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
dt = DecisionTreeRegressor(random_state=2022)
params = {'max_depth':[None,3,5], 
          'min_samples_split':[2,5,10],
          'min_samples_leaf':[1,5,10]}
gcv = GridSearchCV(dt, param_grid=params, scoring='r2',
                          cv = kfold, verbose=3)
gcv.fit(X,y)
print(gcv.best_score_)
print(gcv.best_params_)

######### Grid Search CV : Bagging(Tree)

kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
dt = DecisionTreeRegressor(random_state=2022)
bagging = BaggingRegressor(base_estimator=dt, 
                           random_state=2022)
params = {'base_estimator__max_depth':[None,3,5], 
          'base_estimator__min_samples_split':[2,5,10],
          'base_estimator__min_samples_leaf':[1,5,10],
          'n_estimators':[10,50,100]}
gcv = GridSearchCV(bagging, param_grid=params, scoring='r2',
                          cv = kfold, verbose=3)
gcv.fit(X,y)

pd_cv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_score_)
print(gcv.best_params_)
