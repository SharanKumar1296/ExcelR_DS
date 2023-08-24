#==============================================================================
""" Predicting a model to predict the salary of a person with the help of 
Support Vector Machine Technique """
 
import pandas as pd 
df = pd.read_csv("SalaryData_Train.csv")
df2 = pd.read_csv("SalaryData_Test.csv")

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


#Model fitting using Support Vector Machine technique
from sklearn.svm import SVC
svc = SVC(kernel="linear",C=1)

svc.fit(X_train,Y_train) #Fitting the data to the SVC linear kernel

Y_train_pred = svc.predict(X_train)
Y_test_pred = svc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Training accuracy is",(accuracy_score(Y_train,Y_train_pred)*100).round(2)) #81.24 
print("Testing accuracy is",(accuracy_score(Y_test,Y_test_pred)*100).round(2)) #81.01


svc = SVC(kernel="poly",degree=7)

svc.fit(X_train,Y_train) #Fitting the data to the SVC poly kernel degree 3

Y_train_pred = svc.predict(X_train)
Y_test_pred = svc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Training accuracy is",(accuracy_score(Y_train,Y_train_pred)*100).round(2)) #81.89
print("Testing accuracy is",(accuracy_score(Y_test,Y_test_pred)*100).round(2)) #82.17


svc = SVC(kernel="rbf",gamma=0.5)

svc.fit(X_train,Y_train) #Fitting the data to the SVC poly rbf gamma 0.5

Y_train_pred = svc.predict(X_train)
Y_test_pred = svc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Training accuracy is",(accuracy_score(Y_train,Y_train_pred)*100).round(2)) #85.34
print("Testing accuracy is",(accuracy_score(Y_test,Y_test_pred)*100).round(2)) #82.43

#We have used different kernels and multiple parameters for those kernels and have developed
#the best accuracy scores in accordance with the kernels

#==============================================================================
