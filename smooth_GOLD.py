import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error
from math import sqrt
import nasdaqdatalink
mydata = nasdaqdatalink.get("LBMA/GOLD", authtoken="KUMJYkzkX5gFJxp6MTJJ")
mydata.reset_index(inplace=True)
mydata.plot.line(x='Date', y = 'USD (AM)')
plt.show()


y = mydata['USD (AM)']
y = y.ffill()

y_train = y[:-10]
y_test = y[-10:]

span = 5
#### Trailing MA
fcast = y_train.rolling(span).mean()
MA = fcast.iloc[-1]
MA_series = pd.Series(MA.repeat(len(y_test)))
MA_fcast = pd.concat([fcast,MA_series],ignore_index=True)
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(MA_fcast, label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(y_test, MA_series))
print(rms)


alpha = 0.1
# Simple Exponential Smoothing
fit1 = SimpleExpSmoothing(y_train).fit()
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(y_test, fcast1))
print(rms)


# Holt's Method
alpha = 0.9
beta = 0.02
### Linear Trend
fit1 = Holt(y_train).fit()
fcast1 = fit1.forecast(len(y_test)).rename("Holt's linear trend")

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.show()

rms = sqrt(mean_squared_error(y_test, fcast1))
print(rms)

### Exponential Trend
alpha = 0.9
beta = 0.02
fit2 = Holt(y_train, exponential=True).fit()
fcast2 = fit2.forecast(len(y_test)).rename("Exponential trend")

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast2.plot(color="purple", label='Forecast')
plt.show()
rms = sqrt(mean_squared_error(y_test, fcast2))
print(rms)

### Additive Damped Trend
fit3 = Holt(y_train, damped=True).fit()
fcast3 = fit3.forecast(len(y_test)).rename("Additive damped trend")

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast3.plot(color="purple", label='Forecast')
plt.show()
rms = sqrt(mean_squared_error(y_test, fcast3))
print(rms)

### Multiplicative Damped Trend
fit3 = Holt(y_train,exponential=True, damped=True).fit()
fcast3 = fit3.forecast(len(y_test)).rename("Multiplicative damped trend")

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast3.plot(color="purple", label='Forecast')
plt.show()
rms = sqrt(mean_squared_error(y_test, fcast3))
print(rms)


# Holt-Winters' Method

########### Additive #####################
fit1 = ExponentialSmoothing(y_train, seasonal_periods=4, 
                            trend='add', seasonal='add').fit()

fcast1 = fit1.forecast(len(y_test)).rename("Holt-Winters Additive Trend and seasonality")
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(y_test, fcast1))
print(rms)

########### Multiplicative #####################
fit2 = ExponentialSmoothing(y_train, seasonal_periods=4, trend='add', 
                            seasonal='mul').fit()

fcast2 = fit2.forecast(len(y_test)).rename("Holt-Winters Additive Trend and Multiplicative seasonality")
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast2.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(y_test, fcast2))
print(rms)

########### Seasonal Additive & Damped #####################
fit3 = ExponentialSmoothing(y_train, seasonal_periods=4, trend='add', 
                            seasonal='add', damped=True).fit()

fcast3 = fit3.forecast(len(y_test)).rename("Holt-Winters Additive Trend and seasonality with damping")
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast3.plot(color="purple", label='Forecast')
rms = sqrt(mean_squared_error(y_test, fcast3))
print(rms)

########### Seasonal Multiplicative & Damped #####################
fit4 = ExponentialSmoothing(y_train, seasonal_periods=4, 
                            trend='add', seasonal='mul', 
                            damped=True).fit()

fcast4 = fit4.forecast(len(y_test)).rename("Holt-Winters Multiplicative Trend and Multiplicative seasonality with damping")
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast4.plot(color="purple", label='Forecast')

rms = sqrt(mean_squared_error(y_test, fcast4))
print(rms)

