#=============================================================================
""" Soln.1: Predicting the delivery time from sorting time using Simple Linear
Regression"""  

import numpy as np 
import pandas as pd
df = pd.read_csv("delivery_time.csv")

#Displaying the linearity of the variables
import matplotlib.pyplot as plt
plt.scatter(x=df["Sorting Time"],y=df["Delivery Time"],color="blue")
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

df.corr() #The variables show a strong positive correlation

from sklearn.linear_model import LinearRegression
LR = LinearRegression()

X = df[["Sorting Time"]]
Y = df["Delivery Time"]

LR.fit(X,Y) #Fitted the Linear Regression model using the defined X and Y 

Y_pred = LR.predict(X)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y,Y_pred)
r2 = r2_score(Y,Y_pred)

print("Mean Squared error is",mse.round(2)) #7.79
print("Root Mean Squared error is",np.sqrt(mse).round(2)) #2.79
print("R2 score is",(r2*100).round(2)) #68.23

#As we can see that the R2 score represents a poor model, I will try to better 
#the model using transformation techniques.

#------------------------------------------------------------------------------

#Using log(X) instead of X.

X = df[["Sorting Time"]]
X_log = np.log(X)
Y = df["Delivery Time"]

plt.scatter(x=X_log,y=Y,color="green")
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

LR.fit(X_log,Y) #Fitted the Linear Regression model using log(X) and Y 

Y_pred = LR.predict(X_log)

mse = mean_squared_error(Y,Y_pred)
r2 = r2_score(Y,Y_pred)

print("Mean Squared error is",mse.round(2)) #7.47
print("Root Mean Squared error is",np.sqrt(mse).round(2)) #2.73
print("R2 score is",(r2*100).round(2)) #69.54

#------------------------------------------------------------------------------

#Using X**2 instead of X.

X = df[["Sorting Time"]]
X_sq = X**2
Y = df["Delivery Time"]

plt.scatter(x=X_sq,y=Y,color="yellow")
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

LR.fit(X_sq,Y) #Fitted the Linear Regression model using the defined X**2 and Y 

Y_pred = LR.predict(X_sq)

mse = mean_squared_error(Y,Y_pred)
r2 = r2_score(Y,Y_pred)

print("Mean Squared error is",mse.round(2)) #9.07
print("Root Mean Squared error is",np.sqrt(mse).round(2)) #3.01
print("R2 score is",(r2*100).round(2)) #63.03

#------------------------------------------------------------------------------

#Using sqrt(X) instead of X.

X = df[["Sorting Time"]]
X_sqrt = np.sqrt(X)
Y = df["Delivery Time"]

plt.scatter(x=X_sqrt,y=Y,color="red")
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

LR.fit(X_sqrt,Y) #Fitted the Linear Regression model using sqrt(X) and Y 

Y_pred = LR.predict(X_sqrt)

mse = mean_squared_error(Y,Y_pred)
r2 = r2_score(Y,Y_pred)

print("Mean Squared error is",mse.round(2)) #7.46
print("Root Mean Squared error is",np.sqrt(mse).round(2)) #2.73
print("R2 score is",(r2*100).round(2)) #69.58

#------------------------------------------------------------------------------

#Using 1/sqrt(X) instead of X.

X = df[["Sorting Time"]]
X_sqrt2 = 1/np.sqrt(X)
Y = df["Delivery Time"]

plt.scatter(x=X_sqrt2,y=Y,color="purple")
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

LR.fit(X_sqrt2,Y) #Fitted the Linear Regression model using sqrt(X) and Y 

Y_pred = LR.predict(X_sqrt2)

mse = mean_squared_error(Y,Y_pred)
r2 = r2_score(Y,Y_pred)

print("Mean Squared error is",mse.round(2)) #7.9
print("Root Mean Squared error is",np.sqrt(mse).round(2)) #2.81
print("R2 score is",(r2*100).round(2)) #67.81

""" Despite multiple Transformation Techniques applied on the X variable, i.e,
Sorting Time, we are unable to bring about a drastic shift in the R2_score which
dictates the dependability on the model created. We need to now retrieve more 
sample data with which the model can train and bring about better results for 
predictions. """

#==============================================================================

""" Soln.2: Predicting the Salary from Years of Experience using Simple Linear
Regression""" 


df2 = pd.read_csv("Salary_Data.csv")

#Displaying the linearity of the variables
import matplotlib.pyplot as plt
plt.scatter(x=df2["YearsExperience"],y=df2["Salary"],color="blue")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()

df2.corr() #The variables show a strong positive correlation

from sklearn.linear_model import LinearRegression
LR = LinearRegression()

X = df2[["YearsExperience"]]
Y = df2["Salary"]

LR.fit(X,Y) #Fitted the Linear Regression model using the defined X and Y 

Y_pred = LR.predict(X)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y,Y_pred)
r2 = r2_score(Y,Y_pred)

print("Mean Squared error is",mse.round(2)) #31270951.72
print("Root Mean Squared error is",np.sqrt(mse).round(2)) #5592.04
print("R2 score is",(r2*100).round(2)) #95.7

""" We are able to build a good model for the predictions of Salary. Further 
transformations can be done but the results will not deviate much from the given
error range. """

#==============================================================================