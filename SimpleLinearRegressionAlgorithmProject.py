import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset=pd.read_csv(r'C:\Users\bhupi\OneDrive\Documents\Downloads\Salary_Data.csv')

x=dataset.iloc[:,:-1].values

y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope=regressor.coef_
m_slope

c_inter=regressor.intercept_
c_inter

y_12=9312*12+26780
y_12

y_15=m_slope*15+c_inter
y_15
