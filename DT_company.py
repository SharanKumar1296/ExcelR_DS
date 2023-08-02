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
DTG = DecisionTreeClassifier() #using gini criterion
DTG.fit(X,Y)
Y_pred = DTG.predict(X)
from sklearn.metrics import accuracy_score
print("Accuracy Score is",(accuracy_score(Y,Y_pred)*100).round(2))

DTG.tree_.max_depth #12
DTG.tree_.node_count #141

DTE = DecisionTreeClassifier(criterion="entropy") #using entropy criterion
DTE.fit(X,Y)
Y_pred = DTE.predict(X)

print("Accuracy Score is",(accuracy_score(Y,Y_pred)*100).round(2))

DTE.tree_.max_depth #16
DTE.tree_.node_count #127

#As expected the Decision Tree has overfitted the model 

GTraining_acc = []
GTesting_acc = []
ETraining_acc = []
ETesting_acc = []

from sklearn.model_selection import KFold
kf = KFold(n_splits=5) #Using the K-fold for model validation

for train_index,test_index in kf.split(X):
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    Y_train,Y_test = Y.iloc[train_index],Y.iloc[test_index]
    DTG.fit(X_train,Y_train)
    DTE.fit(X_train,Y_train)
    YG_pred_train = DTG.predict(X_train)
    YG_pred_test = DTG.predict(X_test)
    YE_pred_train = DTE.predict(X_train)
    YE_pred_test = DTE.predict(X_test)
    GTraining_acc.append((accuracy_score(Y_train,YG_pred_train)*100).round(2))
    GTesting_acc.append((accuracy_score(Y_test,YG_pred_test)*100).round(2))
    ETraining_acc.append((accuracy_score(Y_train,YE_pred_train)*100).round(2))
    ETesting_acc.append((accuracy_score(Y_test,YE_pred_test)*100).round(2))

print("Gini Training accuracy Score ",(np.mean(GTraining_acc).round(2)))#100
print("Gini Testing accuracy Score ",(np.mean(GTesting_acc).round(2)))#73.25
print("Entropy Training accuracy Score ",(np.mean(ETraining_acc).round(2)))#100
print("Entropy Testing accuracy Score ",(np.mean(ETesting_acc).round(2)))#71.5

#Using BOOSTING techniques to better the model

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
params = {"n_estimators":[100,150,200],
      "learning_rate":[0.1,0.5,0.8]}

GSV = GridSearchCV(AdaBoostClassifier(random_state=10),param_grid=params) 

GSV.fit(X_train,Y_train)

Y_train_pred = GSV.predict(X_train)
Y_test_pred = GSV.predict(X_test)

print("Training accuracy Score ",(accuracy_score(Y_train,Y_train_pred)*100).round(2))#98
print("Training accuracy Score ",(accuracy_score(Y_test,Y_test_pred)*100).round(2))#75

print("Best model parameters",GSV.best_params_) #200,0.5

#==============================================================================
