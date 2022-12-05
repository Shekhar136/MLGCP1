import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r"C:\Training\Academy\Statistics (Python)\Datasets")

pizza = pd.read_csv("pizza.csv")

sns.scatterplot(x='Promote', y='Sales', data=pizza)
plt.show()

pizza['Promote'].corr(pizza['Sales'])
################# insure_auto ##########################

insure = pd.read_csv("Insure_auto.csv", index_col=0)
insure.corr()

sns.pairplot(insure)
plt.show()

sns.heatmap(
    insure.corr(),
    xticklabels=insure.corr().columns, 
    yticklabels=insure.corr().columns,
    annot=True)
plt.show()

### COncrete

concrete = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength\Concrete_Data.csv")
sns.heatmap(
    concrete.corr(),
    xticklabels=concrete.corr().columns, 
    yticklabels=concrete.corr().columns,
    annot=True)
plt.show()