# ML in Python, homework 3
# Date: 06/19/2019
# Author: Steve G. Mwangi
# Description: Neural network for predicting attrition rate of employees

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Loading the data
# There are 9500 users (rows)
# There are 81 columns for the LIWC features followed by columns for
# openness, conscientiousness, extraversion, agreeableness, neuroticism
# As the target variable, we select the extraversion column (column 83)
dataset = np.loadtxt("Facebook-User-LIWC-personality-HW3.csv", delimiter=",")
X = dataset[:,0:81]
y = dataset[:,83]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1500)

# Training and testing a linear regression model
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)
print('MSE with linear regression:', metrics.mean_squared_error(y_test, y_pred))

# Training and testing a neural network
model = Sequential()

def model_optimization():
    activation_list = ['relu', 'sigmoid','tanh', 'exponential', 'softplus', 'softsign', 'tanh', 'selu', 'elu']
    optimization_list = ['rmsprop', 'sgd', 'adam', 'adagrad', 'adadelta']

    model.add(Dense(16, input_dim=81, kernel_initializer='uniform', activation=activation_list[1]))
    model.add(Dense(8, kernel_initializer='uniform', activation=activation_list[1]))
    model.add(Dense(1, kernel_initializer='uniform'))

    # RMSprop
    # opt = RMSprop(lr=0.1, decay=0.1)
    # model.compile(loss='mse', optimizer=opt, metrics=['mse'])
    model.compile(loss='mse', optimizer=optimization_list[0], metrics=['mse'])

    model.fit(X_train, y_train, epochs=10)
    # model.fit(X_train, y_train, epochs=10)
    y_pred = model.predict(X_test)
    print('MSE with neural network:', metrics.mean_squared_error(y_test, y_pred))

model_optimization()

'''
Original Lines:
model.add(Dense(1,input_dim=81))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MSE with neural network:', metrics.mean_squared_error(y_test, y_pred))
'''