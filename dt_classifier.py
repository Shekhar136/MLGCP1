import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Wisconsin")
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn import tree
import matplotlib.pyplot as plt

cancer = pd.read_csv("BreastCancer.csv", index_col=0)
dum_cancer = pd.get_dummies(cancer, drop_first=True)
X = dum_cancer.drop('Class_Malignant', axis=1)
y = dum_cancer['Class_Malignant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=2022)

clf = DecisionTreeClassifier(max_depth=3,
                             random_state=2022)
clf.fit(X_train,y_train)

plt.figure(figsize=(30,20))
tree.plot_tree(clf,feature_names=X_train.columns,
               class_names=['Benign','Malignant'],
               filled=True,fontsize=20) 

y_pred = clf.predict(X_test)

y_pred_prob = clf.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_pred_prob))

###################### Grid Search CV ###############################
kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=2022)
clf = DecisionTreeClassifier(random_state=2022)
params = {'max_depth':[3,5,7,None], 
          'min_samples_split':[2, 5, 10, 20],
          'min_samples_leaf': [1, 5, 10]}
gcv = GridSearchCV(clf, param_grid=params,
                   scoring='roc_auc', cv=kfold, verbose=3)
gcv.fit(X,y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

plt.figure(figsize=(30,20))
tree.plot_tree(best_model,feature_names=X.columns,
               class_names=['Benign','Malignant'],
               filled=True,fontsize=10) 


############ HR Data #############
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")
hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=2022)
clf = DecisionTreeClassifier(random_state=2022)
params = {'max_depth':[None,10,5,3], 
          'min_samples_split':[2, 10, 50, 100],
          'min_samples_leaf': [1, 10, 50, 100]}
gcv = GridSearchCV(clf, param_grid=params,
                   scoring='roc_auc', cv=kfold, verbose=3)
gcv.fit(X,y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

plt.figure(figsize=(30,20))
tree.plot_tree(best_model,feature_names=X.columns,
               class_names=['Stayed','Left'],
               filled=True,fontsize=10) 


print(best_model.feature_importances_)
ind = np.arange(18)
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns),rotation=90)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()

########## Glass Identification #################
from sklearn.preprocessing import LabelEncoder
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Glass Identification")
glass = pd.read_csv("Glass.csv")

X = glass.drop('Type', axis=1)
y = glass['Type']

le = LabelEncoder()
le_y = le.fit_transform(y)

kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=2022)
clf = DecisionTreeClassifier(random_state=2022)
params = {'max_depth':[None,2,3,4,5,6], 
          'min_samples_split':np.arange(2,11),
          'min_samples_leaf': np.arange(1,11)}
gcv = GridSearchCV(clf, param_grid=params,
                   scoring='neg_log_loss', cv=kfold, verbose=3)
gcv.fit(X,le_y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

print(best_model.feature_importances_)
ind = np.arange(X.shape[1])
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns),rotation=90)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()

############## Brain Stroke ######################
os.chdir(r"C:\Training\Kaggle\Datasets\Brain Stroke")
brain = pd.read_csv("full_data.csv")
dum_brain = pd.get_dummies(brain, drop_first=True)
X = dum_brain.drop('stroke', axis=1)
y = dum_brain['stroke']

kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=2022)
clf = DecisionTreeClassifier(random_state=2022)
params = {'max_depth':[None,3,5,10], 
          'min_samples_split':[2,5,10,50],
          'min_samples_leaf': [1,5,10,50]}
gcv = GridSearchCV(clf, param_grid=params,
                   scoring='roc_auc', cv=kfold, verbose=3)
gcv.fit(X,y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

print(best_model.feature_importances_)
ind = np.arange(X.shape[1])
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns),rotation=90)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()

