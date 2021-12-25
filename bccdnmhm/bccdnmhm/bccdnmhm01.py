import numpy 
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("LatestCovid19IndiaStatus.csv")
data.describe()
print(data)
print(data.keys())
print(data.shape)
print(data.describe())
Y = DataFrame(data, columns=['Discharged']) #tong so ca nhiem covid 19
X = DataFrame(data, columns=['Total Cases'])   #tong so ca khoi covid 19
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=3)
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
print('Hệ số W: \n', regr.coef_) #sai so
print('Hệ số Bias: \n', regr.intercept_) #đo lech, chenh lech gia gia tri tb
Y_pred = regr.predict(X_test)
print('Sai số - bình phương sai số : %.2f '% mean_squared_error(Y_test, Y_pred))
print('Sai số - Hệ số xác định: %.2f'% r2_score(Y_test, Y_pred))
plt.scatter(X, Y, alpha=0.4)
plt.scatter(Y_test, Y_pred)
plt.title('Total Cases vs Discharged in Idian')
plt.xlabel('Total Cases')
plt.ylabel('Discharged')
plt.figure(figsize=(10,6))
plt.scatter(X, Y, alpha=0.4)
plt.plot(X, regression.predict(X), color = 'red', linewidth = 3)
plt.title('Total Cases vs Discharged in Idian')
plt.xlabel('Total Cases')
plt.ylabel('Discharged')
plt.ylim(0, 7000000)
plt.xlim(0, 7000000)
plt.show()
rms = numpy.sqrt(mean_squared_error(Y_test, Y_pred))
print(rms)

