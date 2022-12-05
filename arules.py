import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import os
os.chdir(r'C:\Training\Academy\Statistics (Python)\Association Rules datasets')

fp_df = pd.read_csv('Faceplate.csv',index_col=0)

# Support of 1-tem freq sets
itemFrequency = fp_df.sum(axis=0) 
plt.bar(itemFrequency.index, itemFrequency)
plt.show()

# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.2,use_colnames=True)

# and convert into rules
rules = association_rules(itemsets, metric='confidence', 
                          min_threshold=0.6)

############### Cosmetics #################################
cosmetics = pd.read_csv("Cosmetics.csv", index_col=0)
cosmetics = cosmetics.astype('bool')

# create frequent itemsets
itemsets = apriori(cosmetics, min_support=0.2,use_colnames=True)

# and convert into rules
rules = association_rules(itemsets, metric='confidence', 
                          min_threshold=0.6)

rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

############### Groceries #########################
from mlxtend.preprocessing import TransactionEncoder

groceries = []
with open("groceries.csv","r") as f : groceries = f.read()
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
  
print(groceries_list)

te = TransactionEncoder()
te_ary = te.fit(groceries_list).transform(groceries_list)
te_ary
print(te.columns_)

df = pd.DataFrame(te_ary, columns=te.columns_)
