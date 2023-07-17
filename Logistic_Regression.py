#==============================================================================
""" Predicting the output 'y' using Logistic Regression Technique"""

import numpy as np 
import pandas as pd 

df = pd.read_csv("bank-full.csv",sep=";")

df.shape

df.dtypes 

df_cat = df[["job","marital","education","default","housing","loan","contact",
            "month","poutcome","y"]]


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
 
for i in range(0,11):
    df_cat.iloc[:,i] = LE.fit_transform(df_cat.iloc[:,i])

df[["job","marital","education","default","housing","loan","contact",
    "month","poutcome","y"]] = df_cat

X = df.iloc[:,0:16]
Y = df["y"]

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.model_selection import KFold
KF = KFold(n_splits=5)

Training_acc = []
Testing_acc = []

from sklearn.metrics import accuracy_score

for train_index,test_index in KF.split(X):
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    Y_train,Y_test = Y.iloc[train_index],Y.iloc[test_index]
    logreg.fit(X_train,Y_train)
    Y_pred_train = logreg.predict(X_train)
    Y_pred_test = logreg.predict(X_test)
    Training_acc.append((accuracy_score(Y_train,Y_pred_train)*100).round(2))
    Testing_acc.append((accuracy_score(Y_test,Y_pred_test)*100).round(2))


print("Training Accuracy score",(np.mean(Training_acc)).round(2)) #88.81
print("Testing Accuracy score",(np.mean(Testing_acc)).round(2)) #87.8

#==============================================================================