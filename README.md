# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
 
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

##  data.head()
![image](https://github.com/NaveenKumar-008/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135244/acf71afc-88dc-414e-83b6-7601cbaf5435)

## data.info()
![image](https://github.com/NaveenKumar-008/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135244/f99a8134-a97f-4bd0-8261-6c3918208df9)

## isnull() and sum()
![image](https://github.com/NaveenKumar-008/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135244/366a1964-a0ab-4ec3-a254-e4939291964b)

## data value counts()
![image](https://github.com/NaveenKumar-008/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135244/dc2e792c-08d9-4f23-8bef-8d0f2fd58cb4)

## data.head() for salary
![image](https://github.com/NaveenKumar-008/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135244/310d1e58-dc87-4bdf-8e5c-bd1e8654e832)

## x.head()
![image](https://github.com/NaveenKumar-008/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135244/1485a3a2-d5f8-41d9-83bc-e784550a834f)

## accuracy value
![image](https://github.com/NaveenKumar-008/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135244/3a065d92-bcb9-43a5-a56f-26e9a422f595)

## data prediction
![image](https://github.com/NaveenKumar-008/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135244/f85da82e-6366-4c39-9ad4-a2970fdb2973)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
