import os
os.chdir("G:/Statistics (Python)/Datasets")

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("monthly-milk-production-pounds-p.csv")
df.head()

df.plot()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
series = df['Milk']
result = seasonal_decompose(series, model='additive',freq=12)
print(result.trend)
print(result.seasonal)

print(result.observed)
result.plot()
plt.show()

result = seasonal_decompose(series, model='multiplicative',freq=12)
print(result.trend)
print(result.seasonal)

print(result.observed)
result.plot()
pyplot.show()