import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

### Concrete

concrete = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength\Concrete_Data.csv")



#X = X.values
#X = X.reshape(-1,1)
all_features = list(concrete.columns[:-1])
feature_list = []

for feature in all_features:
    feature_list.append(feature)
    X = concrete[feature_list]
    y = concrete['Strength']
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=2022,
                                                        test_size=0.3)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    print(feature_list,": ",mean_absolute_error(y_test, y_pred))
