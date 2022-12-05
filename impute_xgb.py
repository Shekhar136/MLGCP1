import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Chemical Process Data")
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt

chem = pd.read_csv("ChemicalProcess.csv")

X = chem.drop('Yield', axis=1)
y = chem['Yield']

imputer = SimpleImputer()
gbm = xgb.XGBRegressor(random_state=2022)

pipe = Pipeline([('IMPUTE',imputer),('XGB',gbm)])

kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
params = {'XGB__learning_rate':np.linspace(0.001, 1, 10),
          'XGB__max_depth': [2,3,4,5,6],
          'XGB__n_estimators':[50,100,150],
          'IMPUTE__strategy':['mean','median']}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='r2', verbose=3)

gcv.fit(X, y)

pd_cv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_params_)
print(gcv.best_score_)

################## Iterative Imputer #################
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor

gbm = xgb.XGBRegressor(random_state=2022)
dtr = DecisionTreeRegressor(random_state=2022)
imputer = IterativeImputer(estimator=dtr,max_iter=50,
                           random_state=2022)

pipe = Pipeline([('IMPUTE',imputer),('XGB',gbm)])

kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
params = {'XGB__learning_rate':np.linspace(0.001, 1, 10),
          'XGB__max_depth': [2,3,4,5,6],
          'XGB__n_estimators':[50,100,150]}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='r2', verbose=3)

gcv.fit(X, y)

pd_cv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_params_)
print(gcv.best_score_)
