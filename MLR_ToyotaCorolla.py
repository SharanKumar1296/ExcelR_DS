#==============================================================================
"""Soln:- Creating a prediction model for predicting Price using Multi Linear
Regression Technique """

import numpy as np 
import pandas as pd 
df = pd.read_csv("ToyotaCorolla.csv",encoding="latin1")

df.head()
df.shape

list(df)

Y = df["Price"]
X = df[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
X = pd.DataFrame(mm.fit_transform(X))

from sklearn.linear_model import LinearRegression
LR = LinearRegression()

LR.fit(X,Y)

Y_pred = LR.predict(X)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y,Y_pred)
print("The Error of the model is",(np.sqrt(mse)).round(2)) #1338.26 
r2 = r2_score(Y,Y_pred)
print("The Rsquare score is",(r2*100).round(2)) #86.38

#Validating the above model using validation techniques. Not going to reduce the 
#the model as the problem statement requires us to use the given X variables.

Training_mse = []
Testing_mse = []

from sklearn.model_selection import KFold
kf = KFold(n_splits=5) #Using the K-fold for model validation

for train_index,test_index in kf.split(X):
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    Y_train,Y_test = Y.iloc[train_index],Y.iloc[test_index]
    LR.fit(X_train,Y_train)
    Y_pred_train = LR.predict(X_train)
    Y_pred_test = LR.predict(X_test)
    Training_mse.append(mean_squared_error(Y_train,Y_pred_train))
    Testing_mse.append(mean_squared_error(Y_test,Y_pred_test))
    

print("Average Root Training Error",(np.sqrt(np.mean(Training_mse))).round(2)) #1302.87  
print("Average Root Testing Error",(np.sqrt(np.mean(Testing_mse))).round(2)) #1935.72

#==============================================================================
