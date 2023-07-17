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

#==============================================================================