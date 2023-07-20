#==============================================================================
""" Predicting a model to classify the size of forest fire with the help of 
Support Vector Machine Technique """
 
import pandas as pd 
df = pd.read_csv("forestfires.csv")

df.shape #517x31

df.head()
list(df)

#Acoording to the given set of columns required we are going to drop the unwanted
#columns

df.drop(df.iloc[:,10:30],axis=1,inplace = True)

df.shape #517x11

#Label encoding the required columns 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["month"] = LE.fit_transform(df["month"])
df["day"] = LE.fit_transform(df["day"])
df["size_category"] = LE.fit_transform(df["size_category"])

#Standardizing the remaining columns using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()
df.iloc[:,2:10] = MM.fit_transform(df.iloc[:,2:10])

Y = df["size_category"]
X = df.iloc[:,0:10]

from sklearn.model_selection._split import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=10)


#Model fitting using Support Vector Machine technique
from sklearn.svm import SVC
svc = SVC(kernel="linear",C=1)

svc.fit(X_train,Y_train) #Fitting the data to the SVC linear kernel

Y_train_pred = svc.predict(X_train)
Y_test_pred = svc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Training accuracy is",(accuracy_score(Y_train,Y_train_pred)*100).round(2)) #73.64
print("Testing accuracy is",(accuracy_score(Y_test,Y_test_pred)*100).round(2)) #71.54


svc = SVC(kernel="poly",degree=7)

svc.fit(X_train,Y_train) #Fitting the data to the SVC poly kernel degree 3

Y_train_pred = svc.predict(X_train)
Y_test_pred = svc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Training accuracy is",(accuracy_score(Y_train,Y_train_pred)*100).round(2)) #74.16
print("Testing accuracy is",(accuracy_score(Y_test,Y_test_pred)*100).round(2)) #71.54


svc = SVC(kernel="rbf",gamma=0.5)

svc.fit(X_train,Y_train) #Fitting the data to the SVC poly rbf gamma 0.5

Y_train_pred = svc.predict(X_train)
Y_test_pred = svc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Training accuracy is",(accuracy_score(Y_train,Y_train_pred)*100).round(2)) #74.42
print("Testing accuracy is",(accuracy_score(Y_test,Y_test_pred)*100).round(2)) #70.77

#==============================================================================