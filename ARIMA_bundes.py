import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error
from math import sqrt
#import nasdaqdatalink
#mydata = nasdaqdatalink.get("LBMA/GOLD", authtoken="KUMJYkzkX5gFJxp6MTJJ")
#mydata.reset_index(inplace=True)

mydata = pd.read_csv("BUNDESBANK-BBK01_WT5511.csv")
mydata.plot.line(x='Date', y = 'Value')
plt.show()

y = mydata['Value']

y_train = y[:-12]
y_test = y[-12:]

from pmdarima.arima import auto_arima
model = auto_arima(y_train, trace=True,
                   error_action='ignore', 
                   suppress_warnings=True)

### SARMIA
model = auto_arima(y_train, trace=True, error_action='ignore', 
                   suppress_warnings=True,seasonal=True,m=12)

forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast,index = y_test.index,
                        columns=['Prediction'])

#plot the predictions for validation set
plt.plot(y_train, label='Train',color="blue")
plt.plot(y_test, label='Valid',color="pink")
plt.plot(forecast, label='Prediction',color="purple")
plt.show()


# plot results
plt.plot(y_test)
plt.plot(forecast, color='red')
plt.show()

rms = sqrt(mean_squared_error(y_test, forecast))
print('Test RMSE: %.3f' % rms)

################# Next 5 Predictions ##############
#### Building model on the whole data
model = auto_arima(y, trace=True, error_action='ignore', 
                   suppress_warnings=True)


import numpy as np
forecast = model.predict(n_periods=5)
forecast = pd.DataFrame(forecast,index = np.arange(y.shape[0]+1,y.shape[0]+7),
                        columns=['Prediction'])

#plot the predictions for validation set
plt.plot(y, label='Train',color="blue")

plt.plot(forecast, label='Prediction',color="purple")
plt.show()
 

