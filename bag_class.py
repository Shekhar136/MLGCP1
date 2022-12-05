import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Wisconsin")
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import BaggingClassifier

cancer = pd.read_csv("BreastCancer.csv", index_col=0)
dum_cancer = pd.get_dummies(cancer, drop_first=True)
X = dum_cancer.drop('Class_Malignant', axis=1)
y = dum_cancer['Class_Malignant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=2022)
nb = GaussianNB()
bagging = BaggingClassifier(base_estimator=nb,oob_score=True,
                            random_state=2022)

bagging.fit(X_train, y_train)

print("OOB Score =", bagging.oob_score_)

y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

# Individually using Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_prob = nb.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

### Base est as tree
dt = DecisionTreeClassifier(random_state=2022)
bagging = BaggingClassifier(base_estimator=dt, random_state=2022)

bagging.fit(X_train, y_train)
y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

############# HR Data
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=2022)
#### Naive Bayes
nb = GaussianNB()

# Individually using Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_prob = nb.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

bagging = BaggingClassifier(base_estimator=nb, random_state=2022)

bagging.fit(X_train, y_train)
y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

### Logistic Regression
logreg = LogisticRegression()

# Individually using Logistic

logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

bagging = BaggingClassifier(base_estimator=logreg, random_state=2022)

bagging.fit(X_train, y_train)
y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

#### Decision Tree
dt = DecisionTreeClassifier(random_state=2022)
### Individually
dt.fit(X_train, y_train)
y_pred_prob = dt.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

bagging = BaggingClassifier(base_estimator=dt, random_state=2022)

bagging.fit(X_train, y_train)
y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))


################ Glass Identification ################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Glass Identification")
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

glass = pd.read_csv("Glass.csv")
X = glass.drop('Type', axis=1)
y = glass['Type']

le = LabelEncoder()
le_y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, le_y, test_size=0.3,
                                                    stratify=le_y, random_state=2022)
#### Naive Bayes
nb = GaussianNB()

# Individually using Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_prob = nb.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

bagging = BaggingClassifier(base_estimator=nb, random_state=2022)

bagging.fit(X_train, y_train)
y_pred_prob = bagging.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

### Logistic Regression
logreg = LogisticRegression()

# Individually using Logistic

logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

bagging = BaggingClassifier(base_estimator=logreg, random_state=2022)

bagging.fit(X_train, y_train)
y_pred_prob = bagging.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

#### Decision Tree
dt = DecisionTreeClassifier(random_state=2022)
### Individually
dt.fit(X_train, y_train)
y_pred_prob = dt.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

bagging = BaggingClassifier(base_estimator=dt, random_state=2022)

bagging.fit(X_train, y_train)
y_pred_prob = bagging.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))









