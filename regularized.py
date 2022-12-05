import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

### Concrete

concrete = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength\Concrete_Data.csv")
X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=2022,
                                                    test_size=0.3)
from sklearn.linear_model import Ridge
lr = Ridge(alpha=0.5)
lr.fit(X_train, y_train)
print(lr.intercept_)
print(lr.coef_)
y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))
#### Grid Search CV
params = {'alpha':np.linspace(0.001, 1000)}
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
lr = Ridge()
gcv = GridSearchCV(lr, param_grid=params,scoring='r2',
                          cv = kfold, verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

### Lasso
from sklearn.linear_model import Lasso
params = {'alpha':np.linspace(0.001, 1000)}
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
lr = Lasso()
gcv = GridSearchCV(lr, param_grid=params,scoring='r2',
                          cv = kfold, verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

### ElasticNet
from sklearn.linear_model import ElasticNet
params = {'alpha':np.linspace(0.001, 1000),  'l1_ratio':np.linspace(0, 1)}
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
lr = ElasticNet()
gcv = GridSearchCV(lr, param_grid=params,scoring='r2',
                          cv = kfold, verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

