import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Wisconsin")
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import VotingClassifier

cancer = pd.read_csv("BreastCancer.csv", index_col=0)
dum_cancer = pd.get_dummies(cancer, drop_first=True)
X = dum_cancer.drop('Class_Malignant', axis=1)
y = dum_cancer['Class_Malignant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=2022)

clf = DecisionTreeClassifier(random_state=2022)
nb = GaussianNB()
svm = SVC(probability=True, random_state=2022, kernel='linear')

models = [('Tree', clf),('Naive', nb),('SVM', svm)]

voting = VotingClassifier(models, voting='soft')
voting.fit(X_train, y_train)

y_pred = voting.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

############### HR Data ###########################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)

X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=2022)

clf = DecisionTreeClassifier(random_state=2022)
nb = GaussianNB()
svml = SVC(probability=True, random_state=2022, kernel='linear')
svmr = SVC(probability=True, random_state=2022, kernel='rbf')
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
logreg = LogisticRegression()

models1 = [('Tree',clf),('SVM_Lin', svml),('LDA', lda)]
models2 = [('Tree',clf),('SVM_rbf',svmr),('QDA',qda)]
models3 = [('Logistic',logreg),('Naive',nb),('LDA',lda)]

voting = VotingClassifier(models1, voting='soft')
voting.fit(X_train, y_train)
y_pred_prob = voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

voting = VotingClassifier(models2, voting='soft',verbose=True)
voting.fit(X_train, y_train)
y_pred_prob = voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

voting = VotingClassifier(models3, voting='soft',verbose=True)
voting.fit(X_train, y_train)
y_pred_prob = voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))


