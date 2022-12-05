import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

################## Concrete Strength #####################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")

concrete = pd.read_csv("Concrete_Data.csv")

X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

kfold = KFold(n_splits=5,shuffle=True, random_state=2022)
clf = RandomForestRegressor(random_state=2022)
params = {'max_features':[2,3,4,5,6]}
gcv = GridSearchCV(clf, param_grid=params,
                   scoring='r2', cv=kfold, verbose=3)
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

######### Sorted Plot ################
imp = best_model.feature_importances_

i_sorted = np.argsort(-imp)
col_sorted = X.columns[i_sorted]
imp_sorted = imp[i_sorted]

ind = np.arange(X.shape[1])
plt.bar(ind,imp_sorted)
plt.xticks(ind,(col_sorted),rotation=90)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()




