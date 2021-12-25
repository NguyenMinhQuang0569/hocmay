import numpy
from math import sqrt
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("dtp.csv")
data.describe()
print(data)
print(data.keys())
print(data.shape)
print(data.describe())
Y = DataFrame(data, columns=['worldwide_gross_usd'])
X = DataFrame(data, columns=['production_budget_usd'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
regression = LinearRegression()
regression.fit(X, Y)
regression.coef_
regression.intercept_
regr = linear_model.LinearRegression().fit(X_train, Y_train)
print('Hệ số W: \n', regr.coef_)
print('Hệ số Bias: \n', regr.intercept_)
Y_pred = regr.predict(X_test)
print('Sai số - bình phương sai số : %.2f '% mean_squared_error(Y_test, Y_pred))
print('Sai số - Hệ số xác định: %.2f'% r2_score(Y_test, Y_pred))
plt.scatter(X, Y, alpha=0.3)
plt.scatter(Y_test, Y_pred)
plt.title('film cost vs global revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.figure(figsize=(10,6))
plt.scatter(X, Y, alpha=0.3)
plt.plot(X, regression.predict(X), color = 'red', linewidth = 4)
plt.title('film cost vs global revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(200000000, 450000000)
plt.show()
rms = numpy.sqrt(mean_squared_error(Y_test, Y_pred))
print(rms)
