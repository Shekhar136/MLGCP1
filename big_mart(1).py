import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\AV\Big Mart III")

train = pd.read_csv("train_v9rqX0R.csv")
print(train.columns)
print(train.info())

# Item_Identifier
train['Item_Identifier'].value_counts()

# Item_Weight
train['Item_Weight'].describe()

# Outlet Size
train['Outlet_Size'].value_counts()

# Item_Fat_Content
prev = train['Item_Fat_Content'].value_counts()
train['Item_Fat_Content'].replace({'reg':'Regular',
                                   'LF':'Low Fat',
                                   'low fat':'Low Fat'},
                                  inplace=True)
later = train['Item_Fat_Content'].value_counts()

# Item_Visibility
train['Item_Visibility'].describe()

# Item_Type
train['Item_Type'].value_counts()

# Imputing Item Weights
items = train[['Item_Identifier', 
               'Item_Weight']].sort_values(by='Item_Identifier')

weights_nonmissing = items[items['Item_Weight'].notna()]
weights_nonmissing_nodup = weights_nonmissing.drop_duplicates()
weights_nonmissing_nodup.rename({'Item_Identifier':'Item_Identifier',
                                 'Item_Weight':'i_weight'},
                                axis=1, inplace=True)
train_wt = weights_nonmissing_nodup.merge(train, how='outer',
                                          on='Item_Identifier')

train_wt.drop('Item_Weight', axis=1, inplace=True)

# Imputing Outlet Size
outlets = train[['Outlet_Identifier','Outlet_Establishment_Year',
                'Outlet_Size', 'Outlet_Location_Type','Outlet_Type']]
outlets_nodup = outlets.drop_duplicates()

cnts = train.groupby(['Outlet_Identifier','Outlet_Establishment_Year',
                'Outlet_Size', 'Outlet_Location_Type','Outlet_Type'],
                     dropna=False)['Outlet_Type'].count()

sizes = outlets_nodup[['Outlet_Identifier','Outlet_Size']]
sizes.iloc[2,1] = "Small"
sizes.iloc[5,1] = "Small"
sizes.loc[9,"Outlet_Size"] = "Medium"
sizes.columns = ['Outlet_Identifier','O_Size']

train_wt_out = train_wt.merge(sizes, on="Outlet_Identifier")
train_wt_out.drop('Outlet_Size', axis=1, inplace=True)

#################### Test Set Finding ###########################
test = pd.read_csv("test_AbJTz2l.csv")
# missings in train
train_wt[train_wt['i_weight'].isna()]['Item_Identifier'].unique()
p_test = test[test['Item_Identifier'].isin(['FDN52', 'FDK57', 'FDE52', 'FDQ60'])]
p_test = p_test[['Item_Identifier','Item_Weight']].drop_duplicates().dropna()

####### Back Merging on train set

train_wt_out = train_wt_out.join(p_test.set_index('Item_Identifier'),
                                 on='Item_Identifier')
train_wt_out['i_weight'] = np.where(train_wt_out['i_weight'].isna(),
         train_wt_out['Item_Weight'],
         train_wt_out['i_weight'])

train_wt_out.drop('Item_Weight', axis=1, inplace=True)
print(train_wt_out.info())
########################## EDA #############################################
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from scipy import stats
import seaborn as sns

pearsonr(train_wt_out['i_weight'],
         train_wt_out['Item_Outlet_Sales'])

# ANOVA of Item type and Sales
train_wt_out[['Item_Type','Item_Outlet_Sales']]

aov = ols('Item_Outlet_Sales ~ Item_Type', 
          data=train_wt_out).fit()
table = anova_lm(aov, typ=2)
print(table)

cts = train_wt_out.groupby('Item_Type')['Item_Outlet_Sales'].mean()
cts = cts.sort_values()
plt.barh(cts.index, cts)
plt.show()

# Chi-Square of Item Type and Outlet Size
ctab = pd.crosstab(index=train_wt_out['Item_Type'], 
                   columns=train_wt_out['O_Size'])

test_statistic, p_value, df, expected_frequencies = stats.chi2_contingency(ctab)
print(p_value)
# Conclusion: Any Item Type can be in any outlet size

ctab = ctab.reset_index()
molten = pd.melt(ctab, id_vars='Item_Type',
                 value_name="Count")
sns.catplot(data=molten,
            y='Item_Type', kind="bar",
            x = 'Count', hue="O_Size")
plt.show()

# Chi-Square of Item Type and Outlet Type
ctab = pd.crosstab(index=train_wt_out['Item_Type'], 
                   columns=train_wt_out['Outlet_Type'])

test_statistic, p_value, df, expected_frequencies = stats.chi2_contingency(ctab)
print(p_value)
# Conclusion: Any Item Type can be in any outlet type

ctab = ctab.reset_index()
molten = pd.melt(ctab, id_vars='Item_Type',
                 value_name="Count")
sns.catplot(data=molten,
            y='Item_Type', kind="bar",
            x = 'Count', hue='Outlet_Type')
plt.show()

# ANOVA of Outlet type and Sales
train_wt_out[['Outlet_Type','Item_Outlet_Sales']]

aov = ols('Item_Outlet_Sales ~ Outlet_Type', 
          data=train_wt_out).fit()
table = anova_lm(aov, typ=2)
print(table)

cts = train_wt_out.groupby('Outlet_Type')['Item_Outlet_Sales'].mean()
cts = cts.sort_values()
plt.barh(cts.index, cts)
plt.show()

# ANOVA of Fat Content and Sales
train_wt_out[['Item_Fat_Content','Item_Outlet_Sales']]

aov = ols('Item_Outlet_Sales ~ Item_Fat_Content', 
          data=train_wt_out).fit()
table = anova_lm(aov, typ=2)
print(table)

cts = train_wt_out.groupby('Item_Fat_Content')['Item_Outlet_Sales'].mean()
cts = cts.sort_values()
plt.barh(cts.index, cts)
plt.show()


pearsonr(train_wt_out['Item_MRP'],
         train_wt_out['Item_Outlet_Sales'])

sns.scatterplot(data=train_wt_out,
                x='Item_MRP',y='Item_Outlet_Sales')
plt.show()

############## Buliding a ML Model ####################
X = train_wt_out.drop(['Item_Identifier',
                       'Outlet_Identifier','Item_Outlet_Sales'], axis=1)
y = train_wt_out['Item_Outlet_Sales']

X = pd.get_dummies(X)

import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV
params = {'learning_rate':np.linspace(0.001, 1, 5),
          'max_depth': [2,3,4,5,6],
          'n_estimators':[50,100,150]}
gbm = xgb.XGBRegressor(random_state=2022)
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)

gcv = GridSearchCV(gbm, param_grid=params, cv=kfold,
                   scoring='r2', verbose=3)

gcv.fit(X, y)

pd_cv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

imp=best_model.feature_importances_
i_sort = np.argsort(-imp)
sorted_imp = imp[i_sort]
sorted_col = X.columns[i_sort]

ind = np.arange(X.shape[1])
plt.barh(ind,sorted_imp)
plt.yticks(ind,(sorted_col),fontsize=7)
plt.title('Feature Importance')
plt.ylabel("Variables")
plt.show()

############# Comparing train and test sets #################
items_train = train['Item_Identifier'].unique()
items_test = test['Item_Identifier'].unique()

np.setdiff1d(items_train, items_test)

# Item_Fat_Content
prev = test['Item_Fat_Content'].value_counts()
test['Item_Fat_Content'].replace({'reg':'Regular',
                                   'LF':'Low Fat',
                                   'low fat':'Low Fat'},
                                  inplace=True)
later = test['Item_Fat_Content'].value_counts()


# Imputing Item Weights
items_tst = test[['Item_Identifier', 
               'Item_Weight']].sort_values(by='Item_Identifier')

weights_nonmissing_tst = items_tst[items_tst['Item_Weight'].notna()]
weights_nonmissing_nodup_tst = weights_nonmissing_tst.drop_duplicates()
weights_nonmissing_nodup_tst.rename({'Item_Identifier':'Item_Identifier',
                                 'Item_Weight':'i_weight'},
                                axis=1, inplace=True)

items_trn_tst = weights_nonmissing_nodup_tst.merge(weights_nonmissing_nodup, on="Item_Identifier",
                                how="outer")

items_trn_tst['i_weight'] = np.where(items_trn_tst['i_weight_x'].isna(),items_trn_tst['i_weight_y'],
         items_trn_tst['i_weight_x'])
items_trn_tst = items_trn_tst[['Item_Identifier','i_weight']]
test_wt = items_trn_tst.merge(test, how='right',
                                          on='Item_Identifier')

test_wt['i_weight'] = np.where(test_wt['i_weight'].isna(),test_wt['Item_Weight'],test_wt['i_weight'])
test_wt.drop('Item_Weight', axis=1, inplace=True)


# Imputing Outlet Size
outlets_tst = test[['Outlet_Identifier','Outlet_Establishment_Year',
                'Outlet_Size', 'Outlet_Location_Type','Outlet_Type']]
outlets_nodup_tst = outlets_tst.drop_duplicates()
tst_outlets = outlets_nodup_tst['Outlet_Identifier'].unique()
trn_outlets = outlets_nodup['Outlet_Identifier'].unique()

np.intersect1d(trn_outlets, tst_outlets)

cnts = test.groupby(['Outlet_Identifier','Outlet_Establishment_Year',
                'Outlet_Size', 'Outlet_Location_Type','Outlet_Type'],
                     dropna=False)['Outlet_Type'].count()

sizes = outlets_nodup[['Outlet_Identifier','Outlet_Size']]
sizes.iloc[2,1] = "Small"
sizes.iloc[5,1] = "Small"
sizes.loc[9,"Outlet_Size"] = "Medium"
sizes.columns = ['Outlet_Identifier','O_Size']

test_wt_out = test_wt.merge(sizes, on="Outlet_Identifier")
test_wt_out.drop('Outlet_Size', axis=1, inplace=True)

X_test = test_wt_out.drop(['Item_Identifier',
                            'Outlet_Identifier'], axis=1)

X_test = pd.get_dummies(X_test)
cols_trn = X.columns
cols_tst = X_test.columns

np.setdiff1d(cols_tst, cols_trn)


############ Predicting on test set
y_pred = best_model.predict(X_test)
y_pred[y_pred<0] = 0
test_wt_out['Sales'] = y_pred
test_submit = test_wt_out[['Item_Identifier','Outlet_Identifier',
                           'Sales']]
submit = pd.read_csv("sample_submission_8RXa3c6.csv")

submission = submit.merge(test_submit, on=['Item_Identifier','Outlet_Identifier'])
submission['Item_Outlet_Sales'] = submission['Sales']
submission.drop('Sales', axis=1, inplace=True)

submission.to_csv("XGB_reg.csv", index=False)
