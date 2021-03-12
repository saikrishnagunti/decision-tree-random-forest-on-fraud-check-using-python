# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 22:24:42 2021

@author: shivani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fraud = pd.read_csv("C:\\Users\\shivani\\Desktop\\data science\\module 19 decision tree- random forest\\Fraud_check.csv")
fraud.head(15)
fraud.rename(columns={"Marital.Status":"marital","Taxable.Income":"income","City.Population":"population","Work.Experience":"workexp"},inplace=True)


fraud["taxable"]= "good"
fraud.loc[fraud["income"]<=30000,"taxable"]="risky"

fraud.drop(["income"],axis=1,inplace = True)
fraud["taxable"].unique()
fraud["taxable"].value_counts()

fraud.isnull().sum()

from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
for column_name in fraud.columns:
    if fraud[column_name].dtype == object:
        fraud[column_name] = le.fit_transform(fraud[column_name])
    else:
        pass
    

    
colnames = list(fraud.columns)
type(fraud.columns)
predictors = colnames[:5]
target = colnames[5]


# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(fraud,test_size = 0.2)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors],train[target])

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target],preds,rownames=['Actual'],colnames=['Predictions'])

np.mean(preds==train[target]) # Train Data Accuracy


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target],preds,rownames=['Actual'],colnames=['Predictions'])

np.mean(preds==test[target]) # Test Data Accuracy 



### random forest

X_train = np.array(train.iloc[:,0:4]) # Input
X_test = np.array(test.iloc[:,0:4]) # Input
Y_train = np.array(train['taxable']) # Output
Y_test = np.array(test['taxable']) # Output

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
train_n = norm_func(train)
train_n.describe()

test_n = norm_func(test)
test_n.describe()

from sklearn.ensemble import RandomForestClassifier

rfmodel = RandomForestClassifier(n_estimators=15)

rfmodel.fit(X_train,Y_train)


# Train Data Accuracy
train["rf_pred"] = rfmodel.predict(X_train)
train_acc = np.mean(train["taxable"]==train["rf_pred"])
train_acc

# Test Data Accuracy
test["rf_pred"] = rfmodel.predict(X_test)
test_acc = np.mean(test["taxable"]==test["rf_pred"])
test_acc 
