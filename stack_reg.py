import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Boston Housing")
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

boston = pd.read_csv("boston.csv")

X = boston.drop('medv', axis=1)
y = boston['medv']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=2022)

lr = LinearRegression()
dtr = DecisionTreeRegressor(random_state=2022)
knn = KNeighborsRegressor()
gbm = xgb.XGBRegressor(random_state=2022)
clf = RandomForestRegressor(random_state=2022)

models = [('LIN',lr),('TREE',dtr),('KNN',knn)]
stack = StackingRegressor(estimators=models,final_estimator=clf,
                          passthrough=True)

stack.fit(X_train,y_train)

y_pred = stack.predict(X_test)
print(r2_score(y_test, y_pred))

###### Grid Search ############
lr = LinearRegression()
dtr = DecisionTreeRegressor(random_state=2022)
knn = KNeighborsRegressor()
gbm = xgb.XGBRegressor(random_state=2022)
clf = RandomForestRegressor(random_state=2022)
models = [('LIN',lr),('TREE',dtr),('KNN',knn)]
stack = StackingRegressor(estimators=models,final_estimator=clf,
                          passthrough=True)

params = {'TREE__max_depth':[None, 4],
          'KNN__n_neighbors':[5,7],
          'final_estimator__max_features':['log2','sqrt']}

kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
gcv = GridSearchCV(stack, param_grid=params, cv=kfold)

gcv.fit(X, y)

pd_cv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_params_)
print(gcv.best_score_)


