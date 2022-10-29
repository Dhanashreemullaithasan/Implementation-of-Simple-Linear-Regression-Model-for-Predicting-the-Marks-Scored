# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python.

2.Set variables for assigning dataset values.

3.Import LinearRegression from the sklearn

4.APredict the regression for marks by using the representation of graph.

5.ssign the points for representing the graph.

6.Predict the regression for marks by using the representation of graph.

7.Compare the graphs and hence we obtain the LinearRegression for the given datas.  

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DHANASHREE.M
RegisterNumber: 212221230018

import numpy as np
import pandas as pd
dataset=pd.read_csv("student_scores.csv")
print(dataset.iloc[0:10])
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression() 
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,reg.predict(x_train),color='red')
plt.title('Training set(H vs S) ')
plt.xlabel('Hours')
plt.ylabel('Scores')
y_pred=reg.predict(x_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(x_test,y_test,color='blue')
plt.plot(x_test,reg.predict(x_test),color='red')
plt.title('Test set(H vs S) ')
plt.xlabel('Hours')
plt.ylabel('Scores')
mse=mean_squared_error(y_test,y_pred)
print("MSE ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE ",mae)
rmse=np.sqrt(mse)
print("RMSE ",rmse)

*/
```

## Output:

![M1](https://user-images.githubusercontent.com/94165415/198832571-0eb09088-e21d-4354-80c9-dcf46b3c4b65.png)
![M2](https://user-images.githubusercontent.com/94165415/198832583-09760d28-414b-4791-9667-52eaccf8e94f.png)
![M3](https://user-images.githubusercontent.com/94165415/198832602-ea9d886b-a3e4-40c7-9568-7327d3013b62.png)

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
