import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Boston Housing")
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

boston = pd.read_csv("boston.csv")

X = boston.drop('medv', axis=1)
y = boston['medv']

######### Grid Search CV w/o scaling

kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
sgd = SGDRegressor(random_state=2022)
params = {'eta0':np.linspace(0.001,0.7, 10),
          'learning_rate':['constant','optimal',
                           'invscaling','adaptive']}

gcv = GridSearchCV(sgd, param_grid=params, 
                   scoring='r2',cv=kfold, verbose=3)
gcv.fit(X,y)

print(gcv.best_params_)
print(gcv.best_score_)

######### Grid Search CV with scaling

scaler = MinMaxScaler()
sgd = SGDRegressor(random_state=2022)
pipe = Pipeline([('scl',scaler),('SGD',sgd)])
params = {'SGD__eta0':np.linspace(0.001,0.7, 5),
          'SGD__learning_rate':['constant','optimal',
                           'invscaling','adaptive']}
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
gcv = GridSearchCV(pipe, param_grid=params, 
                   scoring='r2',cv=kfold, verbose=3)
gcv.fit(X,y)

print(gcv.best_params_)
print(gcv.best_score_)

######### Polynomial Features with GridSearch CV
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()
scaler = MinMaxScaler()
sgd = SGDRegressor(random_state=2022)
pipe = Pipeline([('Poly',poly),('scl',scaler),('SGD',sgd)])
params = {'SGD__eta0':np.linspace(0.001,0.7, 5),
          'SGD__learning_rate':['constant','optimal',
                           'invscaling','adaptive'],
          'Poly__degree':[1,2,3,4]}
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
gcv = GridSearchCV(pipe, param_grid=params, 
                   scoring='r2',cv=kfold, verbose=3)
gcv.fit(X,y)

pd_cv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_params_)
print(gcv.best_score_)

################## Concrete Strength #####################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")

concrete = pd.read_csv("Concrete_Data.csv")

X = concrete.drop('Strength', axis=1)
y = concrete['Strength']


######### Polynomial Features with GridSearch CV
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()
scaler = MinMaxScaler()
sgd = SGDRegressor(random_state=2022)
pipe = Pipeline([('Poly',poly),('scl',scaler),('SGD',sgd)])
params = {'SGD__eta0':np.linspace(0.001,0.7,10),
          'SGD__learning_rate':['constant','optimal',
                           'invscaling','adaptive'],
          'Poly__degree':[1,2,3,4]}
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
gcv = GridSearchCV(pipe, param_grid=params, 
                   scoring='r2',cv=kfold, verbose=3)
gcv.fit(X,y)

pd_cv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_params_)
print(gcv.best_score_)


########## Kyphosis
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis")

kyphosis = pd.read_csv("Kyphosis.csv")
dum_kyp = pd.get_dummies(kyphosis, drop_first=True)

X = dum_kyp.drop('Kyphosis_present', axis=1)
y = dum_kyp['Kyphosis_present']

######### Polynomial Features with GridSearch CV
from sklearn.preprocessing import PolynomialFeatures
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
poly = PolynomialFeatures()
scaler = MinMaxScaler()
sgd = SGDClassifier(random_state=2022, loss='log_loss')
pipe = Pipeline([('Poly',poly),('scl',scaler),('SGD',sgd)])
params = {'SGD__eta0':np.linspace(0.001,0.7, 5),
          'SGD__learning_rate':['constant','optimal',
                           'invscaling','adaptive'],
          'Poly__degree':[1,2,3,4]}
gcv = GridSearchCV(pipe, param_grid=params, 
                   scoring='roc_auc',cv=kfold, verbose=3)
gcv.fit(X,y)

pd_cv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_params_)
print(gcv.best_score_)
