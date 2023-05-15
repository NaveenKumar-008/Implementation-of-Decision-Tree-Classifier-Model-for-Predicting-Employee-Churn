# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 
    1.Import the required libraries.
    2.Upload and read the dataset.
    3.Check for any null values using the isnull() function.
    4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
    5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.



## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NAVEEN KUMAR M
RegisterNumber:  212221040113
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:

![O1](https://github.com/22002102/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119091638/cfdb96f6-89e5-4c44-9812-d57cb334c6b7)

![o2](https://github.com/22002102/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119091638/120c2163-9f16-4241-b6c7-0250cc2ea805)


![o3](https://github.com/22002102/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119091638/ddca6407-543d-4d86-921c-4e5fc2dc98ab)


![o4](https://github.com/22002102/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119091638/8de884d1-83e0-42b2-95ee-fae0363f4df0)



![o5](https://github.com/22002102/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119091638/838dba59-a80f-4776-8c69-b760b4b8886d)


![o6](https://github.com/22002102/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119091638/31217d6f-da0e-4866-9787-4f91b4eb8ff4)


![o7](https://github.com/22002102/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119091638/e5628874-4c3d-42d6-991e-906e912327ac)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
