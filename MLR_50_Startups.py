#==============================================================================
"""Creating a prediction model for profit using 50_startups data with the help
of Multi Linear Regression Technique """
  
import numpy as np 
import pandas as pd 
df = pd.read_csv("50_Startups.csv")

df.head() #A quick glance at the data to figure out the datatype and the columns present

#Label Encoding the State column 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["State"] = LE.fit_transform(df["State"])
df["State"]

import matplotlib.pyplot as plt 
plt.scatter(x=df["R&D Spend"],y=df["Profit"],color="green")
plt.xlabel("R&D Spend")
plt.ylabel("Profits")
plt.show()

plt.scatter(x=df["Administration"],y=df["Profit"],color="blue")
plt.xlabel("Administration")
plt.ylabel("Profits")
plt.show()

plt.scatter(x=df["Marketing Spend"],y=df["Profit"],color="yellow")
plt.xlabel("Marketing Spend")
plt.ylabel("Profits")
plt.show()

plt.scatter(x=df["State"],y=df["Profit"],color="orange")
plt.xlabel("State")
plt.ylabel("Profits")
plt.show()

df.corr() #R&D Spend has the highest correlation with the target variable
          #Marketing Spend has the second highest corr followed by Administration
          # and then State. 
         
#We are going to give preference to R&D Spend and Marketing Spend for the prediction
#of our target variable. We can also see R&D Spend and Marketing Spend have high 
#correlation between each other.

Y = df["Profit"]
X1 = df[["R&D Spend"]]
X2 = df[["R&D Spend","Marketing Spend"]]
X3 = df[["R&D Spend","Administration"]]
X4 = df[["R&D Spend","Administration","State"]]
X5 = df[["Marketing Spend"]] 
X6 = df[["Marketing Spend","Administration"]]
X7 = df[["Marketing Spend","Administration","State"]]
X8 = df[["R&D Spend","Marketing Spend","Administration","State"]]

#Using the above created X groups we will fit the data to a Linear Regression 
#model and see the errors along with the R-square scores to obtain the best model.

from sklearn.linear_model import LinearRegression
LR = LinearRegression()

LR.fit(X1,Y) #Fitting the X groups with the target variable. 

Y_pred = LR.predict(X8) #Getting the predicted target variable values from the fit.

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y,Y_pred)
print("The model error is ",np.sqrt(mse).round(2))
r2 = r2_score(Y,Y_pred)
print("The R square score of the model is",(r2*100).round(2))

""" M1(R&D Spend):
    Model Error = 9226.1 , Rsquare = 94.65 """ 
        
""" M2(R&D Spend,Marketing Spend): 
    Model Error = 8881.89 , Rsquare = 95.05 """
        
""" M3(R&D Spend,Administration):
    Model Error = 9115.2 , Rsquare = 94.78 """ 
        
""" M4(R&D Spend,Administration,State): 
    Model Error = 9115.17 , Rsquare = 94.78 """
    
""" M5(Marketing Spend):
    Model Error = 26492.83 , Rsquare = 55.92 """ 
        
""" M6(Marketing Spend,Administration): 
    Model Error = 24927.07 , Rsquare = 60.97 """
        
""" M7(Marketing Spend,Administration,State):
    Model Error = 24874.33 , Rsquare = 61.14 """ 
        
""" M8(R&D Spend,Marketing Spend,Administration,State): 
    Model Error = 8855.33 , Rsquare = 95.07 """
    
#We have 5 models having the R2_score higher than 90, implying these are good models.
#Despite that we need to verify if the correlation between the Xvariables have any 
#negative impacts on the model prediction, hence we need to find the Variance Influence
#factor b/w the Xvariables.

d1 = {"R&D Spend":"RD",
      "Marketing Spend":"Marketing"}
df.rename(columns=d1,inplace="True")

import statsmodels.formula.api as smf
model = smf.ols("RD~Marketing",data=df).fit() #M2
R2 = model.rsquared
VIF = 1/(1-R2)
print("Variance Influence Factor: ",VIF) #2.10

model = smf.ols("RD~Marketing+Administration+State",data=df).fit() #M8
R2 = model.rsquared
VIF = 1/(1-R2)
print("Variance Influence Factor: ",VIF) #2.48

#We can see that there is no multicollinearity amongst the variables with highest
#Rsquare values. Hence we will accept the model M2 as there is very little difference
#in the Rsquare with M8 and uses just two variables.

#==============================================================================