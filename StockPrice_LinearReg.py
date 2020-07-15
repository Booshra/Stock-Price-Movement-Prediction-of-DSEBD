# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 00:42:26 2018

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('stock_prices_bd_2008-2017.csv')

# Removing any data point thats invalid
# dataset = dataset[np.isfinite(dataset['Open'])]
dataset = dataset[dataset.trading_code == 'GLAXOSMITH']
dataset.drop(['trading_code'], 1, inplace=True)
print(dataset.head())

# storing open and close column values in x and y respectively
x = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Sperate train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=0)

# Setting up the model. And that’s pretty much how we create a linear regression model using SciKit.
model = LinearRegression()

# we have to fit this model to our data, in other words
# we have to make it “learn” using our training data. For that, its just one other line of code

model.fit(X_train, y_train)

# Now, my model is trained with the training set I created. We can now start testing the model with the testing dataset we have.
# For that, we add one more line to my code

# yPrediction = model.predict(X_test)

# The next step is to see how well our prediction is working. For this, we’ll use the MatPlotLib library.
# First, we’ll plot the actual values from our dataset against the predicted values for the training set.
# This will tell us how accurate our model is. After that, we’ll make another plot with the test set.
# In both cases, we’ll be using a scatter plot. We’ll plot the actual values (from the dataset) in red, and our model’s predictions in blue.
# This way, we’ll be able to easily differentiate the two.
plot.scatter(X_train, y_train, color='red')
plot.plot(X_train, model.predict(X_train), color='blue')
plot.title('Closing price vs Opening price (Training set)')
plot.xlabel('Opening Price')
plot.ylabel('Closing Price')
plot.savefig('./visualization/Training_set_plot.png', bbox_inches='tight')
plot.show()

plot.scatter(X_test, y_test, color='red')
plot.plot(X_test, model.predict(X_test), color='blue')
plot.title('Closing Price vs Opening Price (Test set)')
plot.xlabel('Opening Price')
plot.ylabel('Closing Price')
plot.savefig('./visualization/Test_set_plot.png', bbox_inches='tight')
plot.show()

# Visualize Results
plot.scatter(X_test, y_test, color='blue', label='Actual Price')  # plotting the initial datapoints
plot.plot(X_train, model.predict(X_train), color='red', linewidth=3,
          label='Predicted Price')  # plotting the line made by linear regression
plot.title('Linear Regression | Time vs. Price')
plot.legend()
plot.xlabel('Date Integer')
plot.savefig('./visualization/Date_plot.png', bbox_inches='tight')
plot.show()

# The mean squared error
print("Mean squared error: %.2f" % np.mean((model.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model.score(X_test, y_test))

accuracy = model.score(X_test, y_test)
print(accuracy * 100, '%')
