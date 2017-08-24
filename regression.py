# -*- coding: utf-8 -*-

import quandl
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = quandl.get('WIKI/AMD')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

df['HL_Change'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low']*100
df['CloseOpen_Change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open']*100

df = df[['Adj. Close', 'HL_Change', 'CloseOpen_Change', 'Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close'
forecast_out = 10

df['label'] = df[forecast_col].shift(-forecast_out)

df.fillna(0, inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.15)

clf = LinearRegression()
clf.fit(X_train, y_train)
print(clf.coef_)
y_predicted = clf.predict(X_test)
error = mean_squared_error(y_test, y_predicted)

print("Mean Squared Error is %s" % error)


