import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Advertising.csv')

feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]

#same as X = data[['TV','Radio','Newspaper']]
#outer bracket is to tell pandas you want to select subset of the dataframe
#inner bracket is how you define a Python list

print(X.head())
print(X.shape)

y = data['Sales']
print(y.head())
print(y.shape)

#Initiating Train Test Split on Data to apply Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
#default split is 75% for training and 25% for testing

linreg = LinearRegression()

linreg.fit(X_train, y_train)

print("Intercept for Linear Regression is:", linreg.intercept_)
print("Coefficient for Linear Regression is:", linreg.coef_)
#underscore is scikitlearn's notation for any attributes that are estimated from the data

zip(feature_cols, linreg.coef_)
#to pair up the values of features and coefficient

# FINAL FORMULA COMES OUT TO BE
# y = 2.876 + 0.0465 x TV + 0.179 x Radio + 0.00345 x Newspaper

y_pred = linreg.predict(X_test)
#print("The accuracy of Linear Regression is :", metrics.accuracy_score(y_test, y_pred))

true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]

#mean absolute error is the mean of absolute errors
mae = (10 + 0 + 20 + 10)/4
print("Mean Absolute Error: ",metrics.mean_absolute_error(true, pred))

#mean squared error is the mean of all squared error values
mse = (10*10 + 0*0 + 20*20 + 10*10)/4
print("Mean Squared Error: ", metrics.mean_squared_error(true, pred))

#root mean squared error is the root of mse
import numpy as np
print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(true, pred)))

#RMSE is most popular cos its in same units as reponse variable y, so easier to interpret

print("RMSE in our Sales Scenario is: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))