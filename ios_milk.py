
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd

milk = pd.read_csv("milk.csv",index_col=0)

clf = IsolationForest(contamination=0.05,random_state=2022)
clf.fit(milk)
predictions = clf.predict(milk)

milk['Outliers'] = predictions
