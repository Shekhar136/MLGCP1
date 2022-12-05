import os
os.chdir("C:/Training/Academy/Statistics (Python)/Datasets")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sktime.forecasting.compose import make_reduction
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.utils.plotting import plot_series

df = pd.read_csv("monthly-milk-production-pounds-p.csv",index_col=0)
idx = pd.to_datetime( df.index ).to_period("M")
df.index = idx

y_train, y_test = temporal_train_test_split(df,test_size=0.1)
fh = np.arange(1, len(y_test) + 1)  # forecasting horizon
regressor = RandomForestRegressor(random_state=2022)
forecaster = make_reduction(regressor,window_length=10)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
rmse = MeanSquaredError(square_root=True)
print(rmse(y_test, y_pred))


plot_series(y_train,y_test,y_pred , labels=['Train','Test','Forecast'])
plt.show()

# plot results
plot_series(y_test,y_pred , labels=['Test','Forecast'])
plt.show()

################ Basic Grid Search CV ####################

from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
)

forecaster = make_reduction(regressor, window_length=15, strategy="recursive")
param_grid = {"window_length": [7, 10, 12, 15]}

# We fit the forecaster on an initial window which is 80% of the historical data
# then use temporal sliding window cross-validation to find the optimal hyper-parameters
cv = SlidingWindowSplitter(initial_window=int(len(y_train) * 0.8), window_length=20)
gscv = ForecastingGridSearchCV(
    forecaster, strategy="refit", cv=cv, param_grid=param_grid
)

gscv.fit(y_train)
y_pred = gscv.predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])

print(gscv.best_params_)
print(gscv.best_score_)

print(rmse(y_test, y_pred))


############## Composite Grid Search CV ###################
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
)

forecaster = make_reduction(regressor, window_length=15, strategy="recursive")
param_grid = {"window_length": [7,10, 12, 15],
              'estimator__max_features':['auto', 'sqrt', 'log2']}
cv = SlidingWindowSplitter(initial_window=int(len(y_train) * 0.8), window_length=20)
gscv = ForecastingGridSearchCV(
    forecaster, strategy="refit", cv=cv, param_grid=param_grid
)

gscv.fit(y_train)
y_pred = gscv.predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])

print(gscv.best_params_)
print(gscv.best_score_)

print(rmse(y_test, y_pred))
plot_series(y_test,y_pred , labels=['Test','Forecast'])
plt.show()