import os
os.chdir(r"C:\Training\Kaggle\Competitions\Bike Sharing Demand")
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

train = pd.read_csv("train.csv", parse_dates=['datetime'])
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['weekday']=train['datetime'].dt.weekday

######## ANOVA ###############

from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
ols_count = ols('count ~ weekday', data=train).fit()
table = anova_lm(ols_count, typ=2)
print(table)

ols_reg = ols('registered ~ weekday', data=train).fit()
table = anova_lm(ols_reg, typ=2)
print(table)

cts = train.groupby('weekday')['registered'].mean()
cts.plot(kind='bar')
plt.show()

ols_cas = ols('casual ~ weekday', data=train).fit()
table = anova_lm(ols_cas, typ=2)
print(table)

cts = train.groupby('weekday')['casual'].mean()
cts.plot(kind='bar')
plt.show()


X = train.drop(['registered','casual','count'], axis=1)
y = train['count']

lr = LinearRegression()
lr.fit(X, y)

test = pd.read_csv("test.csv", parse_dates=['datetime'])
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour
test['weekday']=test['datetime'].dt.weekday

y_pred = lr.predict(test)

### Submission
submit = pd.read_csv("sampleSubmission.csv")

submit['count'] = np.round(y_pred)
print(submit['count'].describe())

y_pred[y_pred<0] = 0

submit['count'] = np.round(y_pred)
print(submit['count'].describe())

########### Separately Predicting ##################

#######################Registered###################
X_test = test.drop('datetime', axis=1)

X = train.drop(['registered','casual',
                'count','datetime'], axis=1)
Registered = train['registered']

lr_registered = LinearRegression()
lr_registered.fit(X, Registered)

pred_registered = lr_registered.predict(X_test)
pred_registered[pred_registered<0] = 0

#######################Causual###################

X = train.drop(['registered','casual',
                'count','datetime'], axis=1)
Casual = train['casual']

lr_casual = LinearRegression()
lr_casual.fit(X, Casual)

pred_casual = lr_casual.predict(X_test)
pred_casual[pred_casual<0] = 0

pred_count = np.round(pred_registered+pred_casual)

### Submission
submit = pd.read_csv("sampleSubmission.csv")

submit['count'] = pred_count