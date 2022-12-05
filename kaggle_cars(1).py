import os
os.chdir(r"C:\Training\Kaggle\Datasets\Cars Prices")
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.linear_model import ElasticNet,Ridge,Lasso
import kaggle

!kaggle datasets download -d hellbuoy/car-price-prediction

cars = pd.read_csv("CarPrice_Assignment.csv",index_col=0)

cars = cars.drop(['CarName'],axis=1)
dum_cars = pd.get_dummies(cars, drop_first=True)
X = dum_cars.drop('price', axis=1)
y = dum_cars['price']

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

### Ridge
params = {'alpha':np.linspace(0.001, 1000)}
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
lr = Ridge()
gcv = GridSearchCV(lr, param_grid=params,scoring='r2',
                          cv = kfold, verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

### Lasso
params = {'alpha':np.linspace(0.001, 1000)}
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
lr = Lasso()
gcv = GridSearchCV(lr, param_grid=params,scoring='r2',
                          cv = kfold, verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

### ElasticNet

params = {'alpha':np.linspace(0.001, 1000),  'l1_ratio':np.linspace(0, 1)}
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
lr = ElasticNet()
gcv = GridSearchCV(lr, param_grid=params,scoring='r2',
                          cv = kfold, verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

### Used Cars
os.chdir(r"C:\Training\Kaggle\Datasets\Used Cars Price Prediction")

!kaggle datasets download -d kukuroo3/used-car-price-dataset-competition-format
