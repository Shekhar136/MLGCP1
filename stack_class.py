import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

brupt = pd.read_csv("Bankruptcy.csv", index_col=0)

X = brupt.drop(['D','YR'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,
                                                    random_state=2022)

lr = LogisticRegression()
svm_l = SVC(random_state=2022, probability=True, kernel='linear')
svm_r = SVC(random_state=2022, probability=True, kernel='rbf')
lda = LinearDiscriminantAnalysis()
dtc = DecisionTreeClassifier(random_state=2022)

clf = RandomForestClassifier(random_state=2022)

models = [('LIN',lr),('SV_LIN',svm_l),('SV_RBF',svm_r),('DA',lda),('TREE',dtc)]
stack = StackingClassifier(estimators=models,final_estimator=clf,
                          passthrough=True)

stack.fit(X_train,y_train)

y_pred_prob = stack.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

###### Grid Search ############
lr = LogisticRegression()
svm_l = SVC(random_state=2022, probability=True, kernel='linear')
svm_r = SVC(random_state=2022, probability=True, kernel='rbf')
lda = LinearDiscriminantAnalysis()
dtc = DecisionTreeClassifier(random_state=2022)

clf = RandomForestClassifier(random_state=2022)

models = [('LIN',lr),('SV_LIN',svm_l),('SV_RBF',svm_r),('DA',lda),('TREE',dtc)]
stack = StackingClassifier(estimators=models,final_estimator=clf,
                          passthrough=True)

params = {'SV_LIN__C':[1,0.5], 
          'TREE__max_depth':[None, 3],
          'final_estimator__max_features':['log2','sqrt']}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
gcv = GridSearchCV(stack, param_grid=params, cv=kfold, scoring='roc_auc', verbose=3)

gcv.fit(X, y)

pd_cv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_params_)
print(gcv.best_score_)






