import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("RidingMowers.csv")

dum_df = pd.get_dummies(df)
dum_df = dum_df.drop('Response_Not Bought', axis=1)

X = dum_df.iloc[:,0:2]
y = dum_df.iloc[:,2]

#### Visualizing the Data
import matplotlib.pyplot as plt
#X_B = X[y==1]
#X_NB = X[y==0]
plt.scatter(X.Income,X.Lot_Size,c="blue")

plt.title("Riding Mowers")
plt.xlabel('Income')
plt.ylabel('Lot Size')
plt.show()

############################################################################

clf = IsolationForest(contamination=0.05,
                      random_state=2022)
clf.fit(X)
predictions = clf.predict(X)



print("%age of outliers="+ str((predictions<0).mean()*100)+ "%")
abn_ind = np.where(predictions < 0)

plt.scatter(X.Income,X.Lot_Size,c="blue",label="Normal Points")
plt.scatter(X['Income'].loc[abn_ind],
            X['Lot_Size'].loc[abn_ind],c="red",label="Outliers")

plt.legend()
plt.title("Riding Mowers")
plt.xlabel('Income')
plt.ylabel('Lot Size')
plt.show()

########################################################
series_outliers = pd.Series(predictions,name="Outliers")
dt_outliers = pd.concat([df,series_outliers],axis=1)
only_outliers = dt_outliers[dt_outliers['Outliers']==-1]
wo_outliers = dt_outliers[dt_outliers['Outliers']!=-1]
