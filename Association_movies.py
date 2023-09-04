#=================================================================================
#pip install mlxtend

import pandas as pd 
df = pd.read_csv("my_movies.csv")
pd.set_option("Display.max_columns",50)

df.head()

df["V1"].value_counts() 
df["V2"].value_counts()  
df["V3"].value_counts()  
df["V4"].value_counts()  
df["V5"].value_counts() 

#On analysing the V1 to V5 columns we can deem them not necessary for the model creation

df.drop(columns=['V1','V2','V3','V4','V5'],inplace=True)
df
df.shape #10x10

from mlxtend.frequent_patterns import apriori,association_rules
#1
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets

frequent_itemsets.shape #53x2

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.6)
rules.shape
list(rules)

rules.sort_values('lift',ascending = False) #Sorted the rules in ascending order 

rules.sort_values('lift',ascending = False)[0:20] #Top 20 of the order

rules[rules.lift>1] #Values with Lift value greater than 1 

rules[['support','confidence','lift']].hist() #Plotting histogram with support,confidence and lift

import matplotlib.pyplot as plt #Scatter plot b/w support and confidence.
plt.scatter(rules['support'], rules['confidence'])
plt.show()

#2
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
frequent_itemsets

frequent_itemsets.shape #7x2

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
rules.shape #12x10
list(rules)

rules.sort_values('lift',ascending = False) #Sorted the rules in ascending order 

rules.sort_values('lift',ascending = False)[0:8] #Top 8 of the order

rules[rules.lift>1] #Values with Lift value greater than 2 

rules[['support','confidence','lift']].hist() #Plotting histogram with support,confidence and lift

import matplotlib.pyplot as plt #Scatter plot b/w support and confidence.
plt.scatter(rules['support'], rules['confidence'])
plt.show()

""" Using the apriori algorithm I was able to establish an association rules relation
and also tried with different values of support,threshold along with plotting the necessary 
graphs """

#=================================================================================