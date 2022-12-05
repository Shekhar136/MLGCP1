import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Boston Housing")
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt

boston = pd.read_csv("boston.csv")

X = boston.drop('medv', axis=1)
y = boston['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=2022)

clf = DecisionTreeRegressor(max_depth=3,
                             random_state=2022)
clf.fit(X_train,y_train)

plt.figure(figsize=(30,20))
tree.plot_tree(clf,feature_names=X_train.columns,
               filled=True,fontsize=20) 

y_pred = clf.predict(X_test)

###################### Grid Search CV ###############################
kfold = KFold(n_splits=5,shuffle=True, random_state=2022)
clf = DecisionTreeRegressor(random_state=2022)
params = {'max_depth':[3,5,7,None], 
          'min_samples_split':[2, 5, 10, 20],
          'min_samples_leaf': [1, 5, 10]}
gcv = GridSearchCV(clf, param_grid=params,
                   scoring='r2', cv=kfold, verbose=3)
gcv.fit(X,y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

plt.figure(figsize=(30,20))
tree.plot_tree(best_model,feature_names=X.columns,
               filled=True,fontsize=10) 

print(best_model.feature_importances_)
ind = np.arange(X.shape[1])
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns),rotation=90)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()

