#=================================================================================
#pip install mlxtend

import pandas as pd 
df = pd.read_csv("bookar.csv")
pd.set_option("Display.max_columns",50)

df.head()
                                #0       #1
df["ChildBks"].value_counts()  #1154     846
df["YouthBks"].value_counts()  #1505     495
df["CookBks"].value_counts()   #1138     862
df["DoItYBks"].value_counts()  #1436     564
df["RefBks"].value_counts()    #1571     429
df["ArtBks"].value_counts()    #1518     482
df["GeogBks"].value_counts()   #1448     552
df["ItalCook"].value_counts()  #1773     227
df["ItalAtlas"].value_counts() #1926      74
df["ItalArt"].value_counts()   #1903      97
df["Florence"].value_counts()  #1783     217

df.shape #2000x11

from mlxtend.frequent_patterns import apriori,association_rules
#1
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets

frequent_itemsets.shape #39x2

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
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
frequent_itemsets = apriori(df, min_support=0.08, use_colnames=True)
frequent_itemsets

frequent_itemsets.shape #60x2

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
rules.shape
list(rules)

rules.sort_values('lift',ascending = False) #Sorted the rules in ascending order 

rules.sort_values('lift',ascending = False)[0:20] #Top 20 of the order

rules[rules.lift>2] #Values with Lift value greater than 2 

rules[['support','confidence','lift']].hist() #Plotting histogram with support,confidence and lift

import matplotlib.pyplot as plt #Scatter plot b/w support and confidence.
plt.scatter(rules['support'], rules['confidence'])
plt.show()

""" Using the apriori algorithm I was able to establish an association rules relation
and also tried with different values of support,threshold along with plotting the necessary 
graphs """

#=================================================================================