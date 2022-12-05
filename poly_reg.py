import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

os.chdir(r"C:\Training\Academy\Statistics (Python)\Datasets")

pizza = pd.read_csv("pizza.csv")

lr = LinearRegression()

X = pizza['Promote']
y = pizza['Sales']

X = X.values
X = X.reshape(-1,1)

poly = PolynomialFeatures(degree=3)
poly_X = poly.fit_transform(X)
print(poly.get_feature_names_out())

pd_poly_X = pd.DataFrame(poly_X, columns=poly.get_feature_names_out())

lr.fit(pd_poly_X, y)
print(lr.intercept_)
print(lr.coef_)

################# insure_auto ##########################

insure = pd.read_csv("Insure_auto.csv", index_col=0)
insure.corr()

X = insure.drop('Operating_Cost', axis=1)
y = insure['Operating_Cost']

poly = PolynomialFeatures(degree=3)
poly_X = poly.fit_transform(X)
print(poly.get_feature_names_out())

pd_poly_X = pd.DataFrame(poly_X, 
                         columns=poly.get_feature_names_out())

lr.fit(pd_poly_X, y)
print(lr.intercept_)
print(lr.coef_)


