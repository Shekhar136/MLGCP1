import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Wisconsin")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

cancer = pd.read_csv("BreastCancer.csv", index_col=0)
dum_cancer = pd.get_dummies(cancer, drop_first=True)
X = dum_cancer.drop('Class_Malignant', axis=1)
y = dum_cancer['Class_Malignant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=2022)
sgd = SGDClassifier(loss='log_loss')
sgd.fit(X_train, y_train)

y_pred_prob = sgd.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

######### Grid Search CV

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
sgd = SGDClassifier(loss='log_loss',random_state=2022)
params = {'eta0':np.linspace(0.001,0.7, 10),
          'learning_rate':['constant','optimal',
                           'invscaling','adaptive']}

gcv = GridSearchCV(sgd, param_grid=params, 
                   scoring='roc_auc',cv=kfold, verbose=3)
gcv.fit(X,y)

print(gcv.best_params_)
print(gcv.best_score_)

####################### Santander
os.chdir(r"C:\Training\Kaggle\Competitions\Santander Customer Satisfaction")

train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)

X = train.drop('TARGET', axis=1)
y = train['TARGET']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
sgd = SGDClassifier(loss='log_loss',random_state=2022)
params = {'eta0':np.linspace(0.001,0.7, 5),
          'learning_rate':['constant','optimal',
                           'invscaling','adaptive']}

gcv = GridSearchCV(sgd, param_grid=params, 
                   scoring='roc_auc',cv=kfold, verbose=3)
gcv.fit(X,y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

y_pred_prob = best_model.predict_proba(test)[:,1]

### Submission 
submit = pd.read_csv("sample_submission.csv")
submit['TARGET'] = y_pred_prob

submit.to_csv("sbt_sgd.csv", index=False)

### MinmAx Scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
scaler = MinMaxScaler()
sgd = SGDClassifier(loss='log_loss',random_state=2022)
pipe = Pipeline([('scl',scaler),('SGD',sgd)])
params = {'SGD__eta0':np.linspace(0.001,0.7, 5),
          'SGD__learning_rate':['constant','optimal',
                           'invscaling','adaptive']}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=2022)
gcv = GridSearchCV(pipe, param_grid=params, 
                   scoring='roc_auc',cv=kfold, verbose=3)
gcv.fit(X,y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

y_pred_prob = best_model.predict_proba(test)[:,1]

### Submission 
submit = pd.read_csv("sample_submission.csv")
submit['TARGET'] = y_pred_prob

submit.to_csv("sbt_sgd.csv", index=False)

########## Otto
os.chdir(r"C:\Training\Kaggle\Competitions\Otto Product Classification")
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv("train.csv", index_col=0)
print(train.shape)

X = train.drop('target', axis=1)
y = train['target']

lbl = LabelEncoder()
le_y = lbl.fit_transform(y)

scaler = MinMaxScaler()
sgd = SGDClassifier(loss='log_loss',random_state=2022)
pipe = Pipeline([('scl',scaler),('SGD',sgd)])
params = {'SGD__eta0':np.linspace(0.001,0.7, 5),
          'SGD__learning_rate':['constant','optimal',
                           'invscaling','adaptive']}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=2022)
gcv = GridSearchCV(pipe, param_grid=params, 
                   scoring='neg_log_loss',cv=kfold, verbose=3)
gcv.fit(X,le_y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

test = pd.read_csv("test.csv", index_col=0)
y_pred_prob = best_model.predict_proba(test)

pd_pred_prob = pd.DataFrame(y_pred_prob,columns=list(lbl.classes_))

### Submission 
submit = pd.read_csv("sampleSubmission.csv")

submission = pd.concat([submit['id'],pd_pred_prob], axis=1)
submission.to_csv("sbt_sgd.csv", index=False)
