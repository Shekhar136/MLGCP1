import h2o
# Starting h2o Engine
h2o.init()
# Loading the data into h2o Dataframe
df = h2o.import_file("C:/Training/Academy/Statistics (Python)/Cases/Bankruptcy/Bankruptcy.csv",
                     destination_frame="Bankruptcy")
print(df.col_names)

y = 'D'
X = df.col_names[3:]

df['D'] = df['D'].asfactor()
print(df['D'].levels())

train,  test = df.split_frame(ratios=[0.7],seed=2022)
print(df.shape)
print(train.shape)
print(test.shape)

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

glm_logistic = H2OGeneralizedLinearEstimator(family = "binomial")

glm_logistic.train(x=X, y= y, training_frame=train, 
                   validation_frame=test, model_id="glm_logistic")

y_pred = glm_logistic.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()

print(glm_logistic.confusion_matrix() )
print(glm_logistic.auc() )

#print(glm_logistic.model_performance())
h2o.cluster().shutdown()

################### Bank ##########################
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\bank")
import pandas as pd

import h2o
# Starting h2o Engine
h2o.init()

bank = pd.read_csv("bank-full.csv", sep=";")
dum_bnk = pd.get_dummies(bank, drop_first=True)

h2o_bnk = h2o.H2OFrame(dum_bnk)
print(h2o_bnk.col_names)

all_columns = h2o_bnk.col_names
X = all_columns[:-1]
y = 'y_yes'

h2o_bnk['y_yes'] = h2o_bnk['y_yes'].asfactor()
print(h2o_bnk['y_yes'].levels())

train,  test = h2o_bnk.split_frame(ratios=[0.7],seed=2022)
print(h2o_bnk.shape)
print(train.shape)
print(test.shape)

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

glm_logistic = H2OGeneralizedLinearEstimator(family = "binomial")

glm_logistic.train(x=X, y= y, training_frame=train, 
                   validation_frame=test, model_id="glm_logistic")

y_pred = glm_logistic.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()

print(glm_logistic.confusion_matrix() )
print(glm_logistic.auc() )

###### Random Forest
from h2o.estimators.random_forest import H2ORandomForestEstimator

rf = H2ORandomForestEstimator(seed=2022)
rf.train(x=X, y= y, training_frame=train, 
                   validation_frame=test, model_id="random_forest")

y_pred = rf.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()

print(rf.confusion_matrix() )
print(rf.auc() )

###### Gradient Boosting
from h2o.estimators.gbm import H2OGradientBoostingEstimator

gbm = H2OGradientBoostingEstimator(seed=2022)
gbm.train(x=X, y= y, training_frame=train, 
                   validation_frame=test, model_id="GBM")

y_pred = gbm.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()

print(gbm.confusion_matrix() )
print(gbm.auc() )
#print(glm_logistic.model_performance())

###########################Tuning with Grid Search#################################
from h2o.grid.grid_search import H2OGridSearch
rf_params1 = { "ntrees" : [10,25,50],
                "max_depth": [ 5, 7, 10],
                "mtries" : [3,4,6,8,10]}

rf_h2o = H2ORandomForestEstimator(seed=2022)
rf_grid1 = H2OGridSearch(model=rf_h2o,
                          grid_id='rf_grid1',
                          hyper_params=rf_params1)
rf_grid1.train(x=X, y=y,training_frame=df,seed=2022)
rf_gridperf1 = rf_grid1.get_grid(sort_by="logloss",
                                 decreasing=False)
rf_gridperf1

best_rf1 = rf_gridperf1.models[0]



h2o.cluster().shutdown()


