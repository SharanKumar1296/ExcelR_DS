#==============================================================================
""" Predicting a model to predict the salary of a person with the help of 
Naive Bayes Classifier """

import pandas as pd 
df = pd.read_csv("SalaryData_TrainNB.csv")
df2 = pd.read_csv("SalaryData_TestNB.csv")

df.shape #30161x14
df2.shape #15060x14

df.head()
df2.head()

df.dtypes
df2.dtypes

obj = ["workclass","education","maritalstatus","occupation","relationship","race",
       "sex","native","Salary"]
std = ["age","educationno","capitalgain","capitalloss","hoursperweek"]

#Label encoding the required columns 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for i in obj:
    df[i] = LE.fit_transform(df[i])
    df2[i] = LE.fit_transform(df2[i])

#Standardizing the remaining columns using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()

for i in std:
    df[[i]] = MM.fit_transform(df[[i]])
    df2[[i]] = MM.fit_transform(df2[[i]])
    
Y_train = df["Salary"]
X_train = df.iloc[:,0:13]

Y_test = df2["Salary"]
X_test = df2.iloc[:,0:13]


#Model fitting using Naive Bayes

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()

MNB.fit(X_train,Y_train)

Y_train_pred = MNB.predict(X_train)
Y_test_pred = MNB.predict(X_test)

from sklearn.metrics import accuracy_score
print("Training Accuracy Score",(accuracy_score(Y_train,Y_train_pred)*100).round(3)) #76.811
print("Test Accuracy Score",(accuracy_score(Y_test,Y_test_pred)*100).round(3)) #76.892

#==============================================================================