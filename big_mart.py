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
