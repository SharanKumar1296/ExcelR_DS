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

#Validating the above model using validation techniques. NOT SHRINKING the 
#model as the problem statement requires us to use the given X variables.

#Using cooks distance to identify outliers 
import statsmodels.formula.api as smf
model = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit()
model.summary()

model_influence = model.get_influence()
(cooks,pvalue) = model_influence.cooks_distance

cooks = pd.DataFrame(cooks)

#Finding the influencers using stem plot 
import matplotlib.pyplot as plt
fig = plt.subplots(figsize=(15,5)) 
plt.stem(np.arange(len(df)),np.round(cooks[0],3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show() #The plot showing all values with their cooks distance

cooks[0][cooks[0]>5] #finds the index which has cooks distance>5

#Plotting the influence plots
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()

k = df.shape[1]
n = df.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff #0.0814

cooks[0][cooks[0]>leverage_cutoff] #five indices identified greater than the leverage cutoff

df.shape
df.drop([80,109,221,601,960],inplace=True) #Dropping the high influence values in accordance with the leverage cutoff 
df.shape

Y = df["Price"]
X = df[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

Training_mse = []
Testing_mse = []
Training_r2 = []
Testing_r2 = []

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
    Training_r2.append(r2_score(Y_train,Y_pred_train))
    Testing_r2.append(r2_score(Y_test,Y_pred_test))    

print("Average Root Training Error",(np.sqrt(np.mean(Training_mse))).round(2)) #1184.29  
print("Average Root Testing Error",(np.sqrt(np.mean(Testing_mse))).round(2)) #1693.24
print("Average Training R-square",((np.mean(Training_r2))*100).round(2)) #85.28  
print("Average Testing R-square",((np.mean(Testing_r2))*100).round(2)) #78.06

#==============================================================================
