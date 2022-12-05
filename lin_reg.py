import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

os.chdir(r"C:\Training\Academy\Statistics (Python)\Datasets")

pizza = pd.read_csv("pizza.csv")

lr = LinearRegression()

X = pizza['Promote']
y = pizza['Sales']

X = X.values
X = X.reshape(-1,1)

lr.fit(X, y)
print(lr.intercept_)
print(lr.coef_)

################# insure_auto ##########################

insure = pd.read_csv("Insure_auto.csv", index_col=0)
insure.corr()

X = insure.drop('Operating_Cost', axis=1)
y = insure['Operating_Cost']

lr.fit(X, y)
print(lr.intercept_)
print(lr.coef_)


### Concrete

concrete = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength\Concrete_Data.csv")

X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=2022,
                                                    test_size=0.3)

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.intercept_)
print(lr.coef_)

y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))

#### K-Folds CV
from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
lr = LinearRegression()
results = cross_val_score(lr, X, y ,scoring='r2',
                          cv = kfold)
print(results)
print(results.mean())

############### Medical Cost Expenses #############
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Medical Cost Personal")
insure = pd.read_csv("insurance.csv")
dum_insure = pd.get_dummies(insure, drop_first=True)

X = dum_insure.drop('charges', axis=1)
y = dum_insure['charges']

scaler = StandardScaler()
knn = KNeighborsRegressor()

pipe = Pipeline([('scaling', scaler),('knn_model',knn)])

print(pipe.get_params())
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
params = {'knn_model__n_neighbors':np.arange(1,31)}

gcv = GridSearchCV(pipe, param_grid=params,scoring='r2',
                          cv = kfold, verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

## Linear Regression
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
lr = LinearRegression()
results = cross_val_score(lr, X, y ,scoring='r2',
                          cv = kfold)
print(results.mean())

### Boston
os.chdir(r"C:\Training\Academy\Statistics (Python)\Datasets")

boston = pd.read_csv("Boston.csv")
X = boston.drop('medv', axis=1)
y = boston['medv']

scaler = StandardScaler()
knn = KNeighborsRegressor()

pipe = Pipeline([('scaling', scaler),('knn_model',knn)])

print(pipe.get_params())
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
params = {'knn_model__n_neighbors':np.arange(1,31)}

gcv = GridSearchCV(pipe, param_grid=params,scoring='r2',
                          cv = kfold, verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

## Linear Regression
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
lr = LinearRegression()
results = cross_val_score(lr, X, y ,scoring='r2',
                          cv = kfold)
print(results.mean())
