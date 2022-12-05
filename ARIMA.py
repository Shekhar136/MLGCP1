import os
#os.chdir("G:/Statistics (Python)/Datasets")

import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error

df = pd.read_csv("AusGas.csv")
df.head()

df.plot.line(x = 'Month',y = 'GasProd')
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['GasProd'], lags=20)
plt.show()

# =============================================================================
# from statsmodels.graphics.tsaplots import plot_pacf
# plot_pacf(df['GasProd'], lags=30)
# plt.show()
# =============================================================================

y = df['GasProd']
y_train = y[:464]
y_test = y[464:]

from statsmodels.tsa.ar_model import AutoReg
# train autoregression
model = AutoReg(y_train, lags=12)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)
    
error = mean_squared_error(y_test, predictions)
print('Test RMSE: %.3f' % sqrt(error))
# plot results
plt.plot(y_test, label='Test')
plt.plot(predictions, color='red', label='Predicted')
plt.legend(loc='best')
plt.show()

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
predictions.plot(color="purple", label='Forcast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(y_test, predictions))
print('Test RMSE: %.3f' % rms)

########################## MA ##############################
from statsmodels.tsa.arima.model import ARIMA
# train MA
model = ARIMA(y_train,order=(0,0,1))
model_fit = model.fit()

print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)
    
error = mean_squared_error(y_test, predictions)
print('Test RMSE: %.3f' % sqrt(error))
# plot results
plt.plot(y_test)
plt.plot(predictions, color='red')
plt.show()

# plot
y_train.plot(color="blue")
y_test.plot(color="pink")
predictions.plot(color="purple")

rms = sqrt(mean_squared_error(y_test, predictions))
print('Test RMSE: %.3f' % rms)

########################## ARMA ##############################
from statsmodels.tsa.arima.model import ARIMA

# train ARMA
model = ARIMA(y_train,order=(7,0,0))
model_fit = model.fit()

print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)
    
error = mean_squared_error(y_test, predictions)
print('Test RMSE: %.3f' % sqrt(error))
# plot results
plt.plot(y_test)
plt.plot(predictions, color='red')
plt.show()

# plot
y_train.plot(color="blue")
y_test.plot(color="pink")
predictions.plot(color="purple")

rms = sqrt(mean_squared_error(y_test, predictions))
print('Test RMSE: %.3f' % rms)

################# ARIMA ####################################

from statsmodels.tsa.arima.model import ARIMA

# train ARIMA
model = ARIMA(y_train,order=(3,1,0))
model_fit = model.fit()

print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)
    
error = mean_squared_error(y_test, predictions)
print('Test RMSE: %.3f' % sqrt(error))

# plot results
plt.plot(y_test)
plt.plot(predictions, color='red')
plt.show()

# plot
y_train.plot(color="blue")
y_test.plot(color="pink")
predictions.plot(color="purple")

rms = sqrt(mean_squared_error(y_test, predictions))
print('Test RMSE: %.3f' % rms)


#############################################################

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

################# Next 6 Months Prediction ##############
#### Building model on the whole data
model = auto_arima(y, trace=True, error_action='ignore', 
                   suppress_warnings=True)


import numpy as np
forecast = model.predict(n_periods=6)
forecast = pd.DataFrame(forecast,index = np.arange(y.shape[0]+1,y.shape[0]+7),
                        columns=['Prediction'])

#plot the predictions for validation set
plt.plot(y, label='Train',color="blue")

plt.plot(forecast, label='Prediction',color="purple")
plt.show()
 

