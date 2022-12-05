import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Medical Cost Personal")
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor

medical = pd.read_csv("insurance.csv")
dum_medical = pd.get_dummies(medical, drop_first=True)
X = dum_medical.drop('charges', axis=1)
y = dum_medical['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=2022)
dtree = DecisionTreeRegressor(random_state=2022)
lr = LinearRegression()
elastic = ElasticNet()

models = [('Elastic',elastic),('LinReg',lr),('Tree',dtree)]
voting = VotingRegressor(models)

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(r2_score(y_test, y_pred))

### Elastic
elastic.fit(X_train, y_train)
y_pred = elastic.predict(X_test)
r2_elastic = r2_score(y_test, y_pred)
print(r2_elastic)

### Linear Regression
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred)
print(r2_lr)

### Decision Tree
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
r2_dtree = r2_score(y_test, y_pred)
print(r2_dtree)

### Weights
voting = VotingRegressor(models, weights=np.array([r2_elastic,r2_lr,r2_dtree]))

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(r2_score(y_test, y_pred))

############### Concrete Strength ############################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")
concrete = pd.read_csv("Concrete_Data.csv")
X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=2022)
dtree = DecisionTreeRegressor(random_state=2022)
lr = LinearRegression()
elastic = ElasticNet()

models = [('Elastic',elastic),('LinReg',lr),('Tree',dtree)]
voting = VotingRegressor(models)

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(r2_score(y_test, y_pred))

### Elastic
elastic.fit(X_train, y_train)
y_pred = elastic.predict(X_test)
r2_elastic = r2_score(y_test, y_pred)
print(r2_elastic)

### Linear Regression
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred)
print(r2_lr)

### Decision Tree
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
r2_dtree = r2_score(y_test, y_pred)
print(r2_dtree)

### Weighted
voting = VotingRegressor(models, weights=np.array([r2_elastic,r2_lr,r2_dtree]))
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(r2_score(y_test, y_pred))

########## Grid Search CV
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
models = [('Elastic',elastic),('LinReg',lr),('Tree',dtree)]
voting = VotingRegressor(models)

params = {'Elastic__alpha':[0.5,1,1.5],
          'Elastic__l1_ratio':[0.25, 0.5, 0.75],
          'Tree__max_depth':[None, 3, 5],
          'Tree__min_samples_split':[2, 5, 10],
          'Tree__min_samples_leaf':[1, 5, 10]}

gcv = GridSearchCV(voting, param_grid=params,
                   scoring='r2',cv=kfold,verbose=3)
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)

######### California Housing ###############
from sklearn.datasets import fetch_california_housing
X_california, y_california = fetch_california_housing(return_X_y=True,
                                                      as_frame=True)
