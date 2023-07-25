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

#Using cooks distance to identify outliers 
import statsmodels.formula.api as smf
model = smf.ols('Profit~RD+Marketing',data=df).fit()
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

cooks[0][cooks[0]>0.1] #finds the index which has cooks distance>0.1

#Plotting the influence plots
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()

k = df.shape[1]
n = df.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff  #0.36

cooks[0][cooks[0]>leverage_cutoff] #one index identified greater than the leverage cutoff

df.shape
df.drop([49],inplace=True) #Dropping the high influence values in accordance with the leverage cutoff 
df.shape

#Using the M2 model to perform validation to explore better results. 

Training_mse = []
Testing_mse = []
Training_r2 = []
Testing_r2 = []

Y = df["Profit"]
X = df[["RD","Marketing"]]

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
    
print("Average Root Training Error",(np.sqrt(np.mean(Training_mse))).round(2)) #7385.5  
print("Average Root Testing Error",(np.sqrt(np.mean(Testing_mse))).round(2)) #8184.34
print("Average Training R-square",((np.mean(Training_r2))*100).round(2)) #95.35  
print("Average Testing R-square",((np.mean(Testing_r2))*100).round(2)) #93.54  

#==============================================================================
