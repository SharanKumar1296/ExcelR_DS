#=================================================================================

import pandas as pd 
import numpy as np
pd.set_option("display.max_columns",50)

df = pd.read_excel("CocaCola_Sales_Rawdata.xlsx")
df

df.shape #42x2

#Extracting the quarters from the Quarter column 
df['quarter'] = 0
for i in range(42):
    p=df['Quarter'][i]
    df['quarter'][i]=p[0:2]
    

df_dummies=pd.DataFrame(pd.get_dummies(df["quarter"]),columns=["Q1","Q2","Q3","Q4"])
df=pd.concat([df,df_dummies],axis= 1)
df
     
import matplotlib.pyplot as plt #Plotting a heatmap of the data at hand
import seaborn as sns
plt.figure(figsize=(12,8))
heatmap_quarters = pd.pivot_table(data=df,values="Sales",index="quarter",fill_value=0)
sns.heatmap(heatmap_quarters,annot=True,fmt="g") 

plt.figure(figsize=(8,6)) #Boxplot for Sales v/s Quarters 
sns.boxplot(x="quarter",y="Sales",data=df)
 
plt.figure(figsize=(12,3)) #Line plot showing the growth through the quarters
sns.lineplot(x="quarter",y="Sales",data=df)

t=np.arange(1,43)
df["t"] = t     
df["t_sq"] = df["t"]*df["t"] 
df["log_Sales"]=np.log(df["Sales"])
df

#Splitting the data into training and test data 
df.shape #42x10
train = df.head(35)
test = df.tail(9)

import statsmodels.formula.api as smf 

# linear model
linear_model = smf.ols("Sales ~ t",data=train).fit()
pred_linear = pd.Series(linear_model.predict(test["t"]))
rmse_linear = np.sqrt(np.mean((np.array(test["Sales"])-np.array(pred_linear))**2))
rmse_linear #641.167

#Exponential
Exp = smf.ols("log_Sales~t",data=train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(test["t"])))
rmse_Exp = np.sqrt(np.mean((np.array(test["Sales"])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp #516.641

#Quadratic 
Quad = smf.ols("Sales~t+t_sq",data=train).fit()
pred_Quad = pd.Series(Quad.predict(test[["t","t_sq"]]))
rmse_Quad = np.sqrt(np.mean((np.array(test["Sales"])-np.array(pred_Quad))**2))
rmse_Quad #469.599

#Additive seasonality 
add_sea = smf.ols("Sales~Q1+Q2+Q3+Q4",data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[["Q1","Q2","Q3","Q4"]]))
rmse_add_sea = np.sqrt(np.mean((np.array(test["Sales"])-np.array(pred_add_sea))**2))
rmse_add_sea #1780.058

#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols("Sales~t+t_sq+Q1+Q2+Q3+Q4",data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[["Q1","Q2","Q3","Q4","t","t_sq"]]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test["Sales"])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad #277.098

##Multiplicative Seasonality
Mul_sea = smf.ols("log_Sales~Q1+Q2+Q3+Q4",data=train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(test["Sales"])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea #1860.125

#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols("log_Sales~t+Q1+Q2+Q3+Q4",data=train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test["Sales"])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea #333.897

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
type(data)

table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

"""The above models have 4 dummy variables created.From the above forecasting methods tried and tested with 
training and test data we can conclude that the additive seasonality quadratic is the best model owing to 
it's low RMSE value.Hence we will use this model for forecasting purposes for the given dataset."""

#=================================================================================
