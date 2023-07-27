#==============================================================================
"""Using Random Forests to prepare a model on fraud data treating those who have 
taxable_income <= 30000 as "Risky" and others are "Good" """

import numpy as np 
import pandas as pd 
df = pd.read_csv("Fraud_check.csv")

#Creating the Y variable according to the given condition 
df["Status"] = df["Taxable.Income"]
df["Status"] = np.where(df["Status"]<=30000,"Risky","Good")
df["Status"].value_counts()

df.shape #600x7

df.dtypes

#Label Encoding the required columns
obj = ("Undergrad","Marital.Status","Urban","Status")

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for i in obj:
    df[i] = LE.fit_transform(df[i])
    
#Data standardizing the required columns 
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

df[["Taxable.Income"]] = mm.fit_transform(df[["Taxable.Income"]])
df[["City.Population"]] = mm.fit_transform(df[["City.Population"]])
df[["Work.Experience"]] = mm.fit_transform(df[["Work.Experience"]])


df.head() #Created a label encoded and standardised dataset 

X = df.iloc[:,0:6]
Y = df["Status"]

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X,Y)
Y_pred = DT.predict(X)
from sklearn.metrics import accuracy_score
print("Accuracy Score is",(accuracy_score(Y,Y_pred)*100).round(2))

DT.tree_.max_depth
DT.tree_.node_count

#We can see that the Decision Tree has created an overfitted model.

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,stratify=Y,random_state=10)

DT = DecisionTreeClassifier()
DT.fit(X_train,Y_train)

Y_train_pred = DT.predict(X_train)
Y_test_pred = DT.predict(X_test)

print("Training accuracy Score ",(accuracy_score(Y_train,Y_train_pred)*100).round(2))#100
print("Training accuracy Score ",(accuracy_score(Y_test,Y_test_pred)*100).round(2))#98.67

#Using RandomForests to better the model

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
params = {"n_estimators":[100,150,200],
          "max_samples":[0.7,0.8],
          "max_features":[0.6,0.7],}
          
GSV = GridSearchCV(RandomForestClassifier(random_state=10),param_grid=params) 

GSV.fit(X_train,Y_train)

Y_train_pred = GSV.predict(X_train)
Y_test_pred = GSV.predict(X_test)

print("Training accuracy Score ",(accuracy_score(Y_train,Y_train_pred)*100).round(2))#100
print("Training accuracy Score ",(accuracy_score(Y_test,Y_test_pred)*100).round(2))#98.67

print("Best model parameters",GSV.best_params_) #Best model parameters {'max_features': 0.6, 'max_samples': 0.7, 'n_estimators': 100}

#==============================================================================

