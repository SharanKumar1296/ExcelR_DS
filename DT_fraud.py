#==============================================================================
"""Using decision trees to prepare a model on fraud data treating those who have 
taxable_income <= 30000 as "Risky" and others are "Good" """

import numpy as np 
import pandas as pd 
df = pd.read_csv("Fraud_check.csv")

#Creating the Y variable according to the given condition 
df["Status"] = df["Taxable.Income"]
df["Status"] = np.where(df["Status"]<=30000,"Risky","Good")
df["Status"].value_counts()

df.shape #600x7

df.dtypes

#Label Encoding the required columns
obj = ("Undergrad","Marital.Status","Urban","Status")

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for i in obj:
    df[i] = LE.fit_transform(df[i])
    
#Data standardizing the required columns 
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

df[["Taxable.Income"]] = mm.fit_transform(df[["Taxable.Income"]])
df[["City.Population"]] = mm.fit_transform(df[["City.Population"]])
df[["Work.Experience"]] = mm.fit_transform(df[["Work.Experience"]])


df.head() #Created a label encoded and standardised dataset 

X = df.iloc[:,0:6]
Y = df["Status"]

from sklearn.tree import DecisionTreeClassifier
DTG = DecisionTreeClassifier() #using gini criterion
DTG.fit(X,Y)
Y_pred = DTG.predict(X)
from sklearn.metrics import accuracy_score
print("Accuracy Score is",(accuracy_score(Y,Y_pred)*100).round(2))

DTG.tree_.max_depth
DTG.tree_.node_count

DTE = DecisionTreeClassifier(criterion="entropy") #using entropy criterion
DTE.fit(X,Y)
Y_pred = DTE.predict(X)

print("Accuracy Score is",(accuracy_score(Y,Y_pred)*100).round(2))

DTE.tree_.max_depth
DTE.tree_.node_count

#We can see that the Decision Tree has created an overfitted model.

GTraining_acc = []
GTesting_acc = []
ETraining_acc = []
ETesting_acc = []

from sklearn.model_selection import KFold
kf = KFold(n_splits=5) #Using the K-fold for model validation

for train_index,test_index in kf.split(X):
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    Y_train,Y_test = Y.iloc[train_index],Y.iloc[test_index]
    DTG.fit(X_train,Y_train)
    DTE.fit(X_train,Y_train)
    YG_pred_train = DTG.predict(X_train)
    YG_pred_test = DTG.predict(X_test)
    YE_pred_train = DTE.predict(X_train)
    YE_pred_test = DTE.predict(X_test)
    GTraining_acc.append((accuracy_score(Y_train,YG_pred_train)*100).round(2))
    GTesting_acc.append((accuracy_score(Y_test,YG_pred_test)*100).round(2))
    ETraining_acc.append((accuracy_score(Y_train,YE_pred_train)*100).round(2))
    ETesting_acc.append((accuracy_score(Y_test,YE_pred_test)*100).round(2))

print("Gini Training accuracy Score ",(np.mean(GTraining_acc).round(2)))#100
print("Gini Testing accuracy Score ",(np.mean(GTesting_acc).round(2)))#99.83
print("Entropy Training accuracy Score ",(np.mean(ETraining_acc).round(2)))#100
print("Entropy Testing accuracy Score ",(np.mean(ETesting_acc).round(2)))#99.83

#==============================================================================
