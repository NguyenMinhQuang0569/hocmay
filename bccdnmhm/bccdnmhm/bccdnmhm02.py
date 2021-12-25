import numpy 
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("USAcoviddata.csv")
data.describe()
print(data)
print(data.keys())
print(data.shape)
print(data.describe())
Y = DataFrame(data, columns=['Total Recovered'])
X = DataFrame(data, columns=['Total Cases'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
lm = LinearRegression()
lm.fit(X_train, Y_train)
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
plt.scatter(X, Y, alpha=0.4)
plt.scatter(Y_test, Y_pred)
plt.title('Total Cases vs Discharged in Idian')
plt.xlabel('Total Cases')
plt.ylabel('Total Recovered')
plt.figure(figsize=(10,6))
plt.scatter(X, Y, alpha=0.3)
plt.plot(X, regression.predict(X), color = 'red', linewidth = 4)
plt.title('Total Cases vs Discharged in Idian')
plt.xlabel('Total Cases')
plt.ylabel('Total Recovered')
plt.ylim(0, 4000000)
plt.xlim(0, 4000000)
plt.show()
rms = numpy.sqrt(mean_squared_error(Y_test, Y_pred))
print(rms)

