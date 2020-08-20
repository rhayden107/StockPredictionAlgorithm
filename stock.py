#  stock.py

import quandl
quandl.ApiConfig.api_key ="H_S7gbwjQei4ARi9Mweo"
import pandas as pd 
import numpy as np 
import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

df = quandl.get("WIKI/TSLA")
df = df[['Adj. Close']]

forecast_out = int(30) # predicting 30 days into future
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out) #  label column with data shifted 30 units up

X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X

y = np.array(df['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Training
clf = LinearRegression()
clf.fit(X_train,y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)

import matplotlib.pyplot as plt
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], forecast_prediction, color="#467ddd", label="Prediction")
plt.title("Tesla (TSLA) 30-Day Stock Prediction")
plt.xlabel(xlabel="Time")
plt.ylabel(ylabel="Price")
plt.legend(loc=3, title=f"Confidence = {confidence}")
plt.savefig("stock_prediction.png")
