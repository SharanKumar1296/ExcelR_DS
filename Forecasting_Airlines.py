#=================================================================================

import pandas as pd 
import numpy as np
pd.set_option("display.max_columns",50)

df = pd.read_excel("Airlines+Data.xlsx")
df

df["month"] = df.Month.dt.strftime("%b") # month extraction
df["year"] = df.Month.dt.strftime("%Y") # year extraction
df.head()

df_dummies=pd.DataFrame(pd.get_dummies(df["month"]))
df=pd.concat([df,df_dummies],axis= 1)
df.head()

     
import matplotlib.pyplot as plt #Plotting a heatmap of the data at hand
import seaborn as sns
plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=df,values="Passengers",index="year",columns="month",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") 

plt.figure(figsize=(8,6)) #Boxplot for month v/s passengers 
sns.boxplot(x="month",y="Passengers",data=df)
 
plt.figure(figsize=(8,6)) #Boxplot for year v/s passengers
sns.boxplot(x="year",y="Passengers",data=df)

plt.figure(figsize=(12,3)) #Line plot showing the growth through the years
sns.lineplot(x="year",y="Passengers",data=df)


t=np.arange(1,97)
df["t"] = t     
df["t_sq"] = df["t"]*df["t"] 
df["log_Passengers"]=np.log(df["Passengers"])
df

#Splitting the data into training and test data 
df.shape #96x2
train = df.head(74)
test = df.tail(22)

import statsmodels.formula.api as smf 

# linear model
linear_model = smf.ols("Passengers ~ t",data=train).fit()
pred_linear = pd.Series(linear_model.predict(test["t"]))
rmse_linear = np.sqrt(np.mean((np.array(test["Passengers"])-np.array(pred_linear))**2))
rmse_linear #55.163

#Exponential
Exp = smf.ols("log_Passengers~t",data=train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(test["t"])))
rmse_Exp = np.sqrt(np.mean((np.array(test["Passengers"])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp #44.412

#Quadratic 
Quad = smf.ols("Passengers~t+t_sq",data=train).fit()
pred_Quad = pd.Series(Quad.predict(test[["t","t_sq"]]))
rmse_Quad = np.sqrt(np.mean((np.array(test["Passengers"])-np.array(pred_Quad))**2))
rmse_Quad #55.825

#Additive seasonality 
add_sea = smf.ols("Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec",data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]]))
rmse_add_sea = np.sqrt(np.mean((np.array(test["Passengers"])-np.array(pred_add_sea))**2))
rmse_add_sea #129.835

#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols("Passengers~t+t_sq+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec",data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","t","t_sq"]]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test["Passengers"])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad #38.479

##Multiplicative Seasonality
Mul_sea = smf.ols("log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec",data=train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(test["Passengers"])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea #134.986

#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols("log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec",data=train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test["Passengers"])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea #11.782

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
type(data)

table_rmse=pd.DataFrame(data)
table_rmse.sort_values(["RMSE_Values"])

"""The above models have 12 dummy variables created.From the above forecasting methods tried and tested with 
training and test data we can conclude that the multiplicative additive seasonality is the best model owing to 
it's low RMSE value.Hence we will use this model for forecasting purposes for the given dataset."""

#=================================================================================

