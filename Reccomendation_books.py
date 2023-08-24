#====================================================================================

import pandas as pd
import numpy as np 
pd.set_option("Display.max_columns",10)

df = pd.read_csv("book.csv",encoding="latin-1")

df.shape #10000x4

df.head()

#As we are interested in the row numbers we can remove the first column which is unnamed

books = df.iloc[:,1:]

#We have a created a new dataframe with the necessary columns

books.describe()

len(books["User.ID"].unique()) #There are 2182 unique IDs from the dataset

len(books["Book.Title"].unique()) #There are 9659 unique book Titles from the dataset

books["Book.Rating"].value_counts()
#8     2283
#7     2076
#10    1732
#9     1493
#5     1007
#6      920
#4      237
#3      146
#2       63
#1       43

books["Book.Rating"].hist()
#From the visual we can easily conclude that majority of the ratings are 5 and above and a 
#small chunk of the ratings lie on the lower half of the scale rating

books[books.duplicated()]
books.drop([5051,7439],axis=0,inplace=True)
books[books.duplicated()]
#We have dropped the duplicate entries from the dataset

#Created a new dataframe with the "userID" as rows and columns as "BookTitle"
user_df = books.pivot_table(index="User.ID",columns="Book.Title",values="Book.Rating") 

user_df.fillna(0,inplace=True) #Removing all NaN values and replacing with 0

from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances(user_df.values,metric='cosine')
user_sim

#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)


#Set the index and column names to user ids 
user_sim_df.index   = books["User.ID"].unique()
user_sim_df.columns = books["User.ID"].unique()

user_sim_df

np.fill_diagonal(user_sim, 0) #nullifying the diagonal values to 0 

#Most Similar Users
user_sim_df.max()

user_sim_df.idxmax(axis=1)[0:100]

books[(books["User.ID"]==276744)|(books["User.ID"]==276726)]

#We have created the recommendation system based on the cosine similiarity and hence 
#can be used for recommending books based off of the cosine distance as shown above.

#=====================================================================================