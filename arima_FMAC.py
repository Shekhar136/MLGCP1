import os
#os.chdir("G:/Statistics (Python)/Datasets")

import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima

df = pd.read_csv("FMAC-HPI_24420.csv")

y = df['NSA Value']

################# Next 4 Months Prediction ##############
#### Building model on the whole data
model = auto_arima(y, trace=True, error_action='ignore', 
                   suppress_warnings=True)

forecast = model.predict(n_periods=4)
#forecast = pd.DataFrame(forecast,index = np.arange(y.shape[0],y.shape[0]+4),
#                        columns=['Prediction'])

#plot the predictions for validation set
plt.plot(y, label='Train',color="blue")
plt.plot(forecast, label='Prediction',color="purple")
plt.show()
 

