# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
# Placement data
![image](https://github.com/user-attachments/assets/3f736cf6-8fd6-4d1e-bed4-9579777714b8)
# Salary data
![image](https://github.com/user-attachments/assets/5fcd898b-f2bb-4465-b76e-1ef902d4c7de)
# Checking null function
![image](https://github.com/user-attachments/assets/4c4b52ac-d1a4-4066-9f03-f76a60990ffb)
# Data duplicate
![image](https://github.com/user-attachments/assets/53ac73a8-cd3b-4ead-a7c9-4a148e1484ed)
# Print data
![image](https://github.com/user-attachments/assets/a54bedf5-194a-4144-b93b-622f47915418)
# Data status
![image](https://github.com/user-attachments/assets/fe74ffef-7c77-4353-b4b1-945febc2ce17)
# Y-prediction array
![image](https://github.com/user-attachments/assets/05862a3d-e717-4d86-8f1c-585d9cbccc29)
# Accuracy value
![image](https://github.com/user-attachments/assets/000f19f4-153f-4af2-8d1a-f6ad61a79d2b)
# Confusion array
![image](https://github.com/user-attachments/assets/0ee3d663-7857-472f-b600-2bc1fd3f95b3)
# Classification report
![image](https://github.com/user-attachments/assets/293cc3b1-5726-464b-aa78-d9e5b5f48e50)
# Prediction of LR
![image](https://github.com/user-attachments/assets/f16a6a4c-afe3-450f-93e3-ab9772f6d561)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
