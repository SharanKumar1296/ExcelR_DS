#==============================================================================
""" A cloth manufacturing company is interested to know about the segment or 
attributes causes high sale. Decision Tree is built with target variable Sale """

import numpy as np 
import pandas as pd 
df = pd.read_csv("Company_Data.csv")
pd.set_option("Display.max_columns",50)

df.shape #400x11

df["Sales"].mean() #Finding the average value to classify the Sales as High and Low 

df["Sales"] = np.where(df["Sales"]<=7.50,"Low","High")
df["Sales"].value_counts() #High:198 Low:202

df.dtypes

#Label Encoding the required columns
obj = ("Sales","ShelveLoc","Urban","US")

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for i in obj:
    df[i] = LE.fit_transform(df[i])
    
#Data standardizing the required columns 
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

df[["CompPrice"]] = mm.fit_transform(df[["CompPrice"]])
df[["Income"]] = mm.fit_transform(df[["Income"]])
df[["Advertising"]] = mm.fit_transform(df[["Advertising"]])
df[["Population"]] = mm.fit_transform(df[["Population"]])
df[["Price"]] = mm.fit_transform(df[["Price"]])
df[["Age"]] = mm.fit_transform(df[["Age"]])
df[["Education"]] = mm.fit_transform(df[["Education"]])


X = df.iloc[:,1:11]
Y = df["Sales"]

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X,Y)
Y_pred = DT.predict(X)
from sklearn.metrics import accuracy_score
print("Accuracy Score is",(accuracy_score(Y,Y_pred)*100).round(2))
#As expected the model has been overfitted

DT.tree_.max_depth #12
DT.tree_.node_count #141

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,stratify=Y,random_state=10)

DT = DecisionTreeClassifier()
DT.fit(X_train,Y_train)

Y_train_pred = DT.predict(X_train)
Y_test_pred = DT.predict(X_test)

print("Training accuracy Score ",(accuracy_score(Y_train,Y_train_pred)*100).round(2))#100
print("Training accuracy Score ",(accuracy_score(Y_test,Y_test_pred)*100).round(2))#68

#Using Random Forests to better the model

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
params = {"n_estimators":[100,150,200],
          "max_samples":[0.7,0.8],
          "max_features":[0.6,0.7,0.8],
          "max_depth":[12,10,8,5]}

GSV = GridSearchCV(RandomForestClassifier(random_state=10),param_grid=params) 

GSV.fit(X_train,Y_train)

Y_train_pred = GSV.predict(X_train)
Y_test_pred = GSV.predict(X_test)

print("Training accuracy Score ",(accuracy_score(Y_train,Y_train_pred)*100).round(2))#99.33
print("Training accuracy Score ",(accuracy_score(Y_test,Y_test_pred)*100).round(2))#75

print("Best model parameters",GSV.best_params_) #Best model parameters {'max_depth': 8, 'max_features': 0.8, 'max_samples': 0.8, 'n_estimators': 150}

#==============================================================================

