import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Wisconsin")
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

cancer = pd.read_csv("BreastCancer.csv", index_col=0)
dum_cancer = pd.get_dummies(cancer, drop_first=True)
X = dum_cancer.drop('Class_Malignant', axis=1)
y = dum_cancer['Class_Malignant']

######### Grid Search CV : RBF
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
svm = SVC(probability=True, kernel='rbf')
params = {'C': np.linspace(0.001, 5, 10),
          'gamma':np.linspace(0.001, 5, 10)}
gcv = GridSearchCV(svm, param_grid=params, scoring='roc_auc',
                          cv = kfold, verbose=3)
gcv.fit(X,y)
print(gcv.best_score_)
print(gcv.best_params_)

######### Grid Search CV : Polynomial
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
svm = SVC(probability=True, kernel='poly')
params = {'C': np.linspace(0.001, 5, 10),
          'degree':[1,2,3]}
gcv = GridSearchCV(svm, param_grid=params, scoring='roc_auc',
                          cv = kfold, verbose=3)
gcv.fit(X,y)
print(gcv.best_score_)
print(gcv.best_params_)

############### Satellite Imaging #####################
from sklearn.preprocessing import LabelEncoder
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Satellite Imaging")
satellite = pd.read_csv("Satellite.csv", sep=";")

X = satellite.drop('classes', axis=1)
y = satellite['classes']

lbl = LabelEncoder()
le_y = lbl.fit_transform(y)

######### Grid Search CV : Linear
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
svm = SVC(probability=True, kernel='linear')
params = {'C': np.linspace(0.001, 3, 5),
          'decision_function_shape':['ovo','ovr']}
gcv = GridSearchCV(svm, param_grid=params, scoring='neg_log_loss',
                          cv = kfold, verbose=3)
gcv.fit(X,le_y)
print(gcv.best_score_)
print(gcv.best_params_)

######### Grid Search CV : RBF
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
svm = SVC(probability=True, kernel='rbf')
params = {'C': np.linspace(0.001, 3, 5),
          'gamma':np.linspace(0.001, 5, 10),
          'decision_function_shape':['ovo','ovr']}
gcv = GridSearchCV(svm, param_grid=params, scoring='neg_log_loss',
                          cv = kfold, verbose=3)
gcv.fit(X,y)
print(gcv.best_score_)
print(gcv.best_params_)

############### Glass #####################
from sklearn.preprocessing import LabelEncoder
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Glass Identification")
glass = pd.read_csv("Glass.csv")

X = glass.drop('Type', axis=1)
y = glass['Type']

lbl = LabelEncoder()
le_y = lbl.fit_transform(y)

######### Grid Search CV : Linear
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
svm = SVC(probability=True, kernel='linear',random_state=2022)
params = {'C': np.linspace(0.001, 5, 10),
          'decision_function_shape':['ovo','ovr']}
gcv = GridSearchCV(svm, param_grid=params, scoring='neg_log_loss',
                          cv = kfold, verbose=3)
gcv.fit(X,le_y)
print(gcv.best_score_)
print(gcv.best_params_)

######### Grid Search CV : RBF
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
svm = SVC(probability=True, kernel='rbf',random_state=2022)
params = {'C': np.linspace(0.001, 5, 10),
          'gamma':np.linspace(0.001, 5, 10),
          'decision_function_shape':['ovo','ovr']}
gcv = GridSearchCV(svm, param_grid=params, scoring='neg_log_loss',
                          cv = kfold, verbose=3)
gcv.fit(X,le_y)
print(gcv.best_score_)
print(gcv.best_params_)

#### Min Max Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

scaler = StandardScaler()
## linear
svm = SVC(probability=True, kernel='linear',random_state=2022)
pipe = Pipeline([('scl',scaler),('SVM',svm)])
params = {'SVM__C': np.linspace(0.001, 5, 10),
          'SVM__decision_function_shape':['ovo','ovr']}
gcv = GridSearchCV(pipe, param_grid=params, scoring='neg_log_loss',
                          cv = kfold, verbose=3)
gcv.fit(X,le_y)
print(gcv.best_score_)
print(gcv.best_params_)

## Radial
svm = SVC(probability=True, kernel='rbf',random_state=2022)
pipe = Pipeline([('scl',scaler),('SVM',svm)])
params = {'SVM__C': np.linspace(0.001, 5, 10),
          'SVM__gamma':np.linspace(0.001, 5, 10),
          'SVM__decision_function_shape':['ovo','ovr']}
gcv = GridSearchCV(pipe, param_grid=params, scoring='neg_log_loss',
                          cv = kfold, verbose=3)
gcv.fit(X,le_y)
print(gcv.best_score_)
print(gcv.best_params_)

