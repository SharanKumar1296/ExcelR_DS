#=============================================================================
""" Creating a KNN model to classify animals into categories """

import numpy as np 
import pandas as pd 
df = pd.read_csv("Zoo.csv")
pd.set_option("display.max_columns",50)

df.shape #101 rows and 18 columns 
df.head()
list(df)

#Label encoding the "animal name" column 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["animal name"] = LE.fit_transform(df["animal name"])

Y = df["type"]
X = df.iloc[:,0:17]

#Transforming the X variables using MinMaxScaler method
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
X = pd.DataFrame(mm.fit_transform(X),columns=list(X))

#Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=10) 

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)

KNN.fit(X_train,Y_train)

Y_train_pred = KNN.predict(X_train)
Y_test_pred = KNN.predict(X_test)

from sklearn.metrics import accuracy_score
print("Training Accuracy Score",(accuracy_score(Y_train, Y_train_pred)*100).round(2)) #96
print("Testing Accuracy Score",(accuracy_score(Y_test, Y_test_pred)*100).round(2)) #88.16

#Using the cross validation technique to deliver a better accuracy score
Training_acc = []
Testing_acc = []

from sklearn.model_selection import KFold
kf = KFold(n_splits=5) #Using the K-fold for model validation

for train_index,test_index in kf.split(X):
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    Y_train,Y_test = Y.iloc[train_index],Y.iloc[test_index]
    KNN.fit(X_train,Y_train)
    Y_pred_train = KNN.predict(X_train)
    Y_pred_test = KNN.predict(X_test)
    Training_acc.append(accuracy_score(Y_train,Y_pred_train))
    Testing_acc.append(accuracy_score(Y_test,Y_pred_test))
   

print("Average Training Accuracy",(np.mean(Training_acc)*100).round(2)) #97.28 
print("Average Testing Accuracy",(np.mean(Testing_acc)*100).round(2)) #91.0
#==============================================================================