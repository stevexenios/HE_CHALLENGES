# ML in Python, homework 3
# Date: 06/19/2019
# Author: Steve G. Mwangi
# Description: Neural network for predicting attrition rate of employees

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
# from utilities import data_prep
from scipy.interpolate import PchipInterpolator, pchip_interpolate


# Displaying all columns
from tensorflow import keras

pd.set_option('display.max_columns', None)


# Method to factorize data frame, by given list
def factorize_df(the_df, the_list):
    label = LabelEncoder()
    for i in the_list:
        # the_df[i], cats = pd.factorize(the_df[i])
        # the_df[i] = pd.Categorical(the_df[i], categories=np.arange(len(cats)))
        the_df[i] = label.fit_transform(the_df[i])
    return the_df


# Function to fill NA values in the columns
def filling_na(the_df, the_list):
    for i in the_list:
        the_df[i] = the_df[i].fillna((the_df[i].median()))
    return the_df


# Function to calculate my score from SQRD
def print_my_score(sq_value):
    score = 100 * (1-sq_value)
    print("The score is: ", score)


# Function to create a csv file from data
def create_csv_file(y_id, y_predi, str_name):
    # Initialize data of lists
    result = {'Employee_ID': y_id, 'Attrition_rate': y_predi}
    result_df = pd.DataFrame(result)

    # Drop nameless column
    # result_df.drop(result_df.columns[result_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    result_df.to_csv(str(str_name + '.csv'), index=False)


# Function to normalize the data in the columns
def normalize_columns(df):
    columns = list(df)
    columns.remove('Employee_ID')
    columns.remove('Attrition_rate')
    for i in columns:
        x = df[[i]].values.astype(float)
        # print(x)
        # Create a minimum and maximum processor object
        min_max_scaler = preprocessing.MinMaxScaler()

        # Create an object to transform the data to fit minmax processor
        x_scaled = min_max_scaler.fit_transform(x)

        # Run the normalizer on the dataframe
        df[[i]] = pd.DataFrame(x_scaled)
    return df


# The is the Linear Regression Model
def training_linear_regression(Xx_train, yy_train, Xx_test, yy_test, X_output, y_ids):
    # Training and testing a linear regression model
    linreg = LinearRegression()
    linreg.fit(Xx_train, yy_train)

    y_pred = linreg.predict(Xx_test)
    sqrd_err = metrics.mean_squared_error(yy_test, y_pred)
    print('\nMSE with linear regression:', sqrd_err)
    print_my_score(sqrd_err)

    # Linear Regression Output
    output_predicted_1 = linreg.predict(X_output)
    print("Output 1: ", output_predicted_1)
    create_csv_file(y_ids, output_predicted_1, 'linear_regression_output')


# Training and testing a neural network
def training_neural_network(Xx_train, yy_train, Xx_test, yy_test, X_output, y_ids):
    model = Sequential()
    activation_list = ['relu', 'sigmoid', 'tanh', 'exponential', 'softplus', 'softsign', 'selu', 'elu']
    optimization_list = ['rmsprop', 'sgd', 'adam', 'adagrad', 'adadelta']

    model.add(Dense(64, input_dim=21, kernel_initializer='uniform', activation=activation_list[1]))
    # model.add(Dense(32, activation=activation_list[1]))
    # model.add(Dense(16, activation=activation_list[1]))
    model.add(Dense(8, kernel_initializer='uniform', activation=activation_list[1]))
    model.add(Dense(1, kernel_initializer='uniform'))

    # RMSprop
    # opt = RMSprop(lr=0.1, decay=0.1)
    # model.compile(loss='mse', optimizer=opt, metrics=['mse'])
    opt = keras.optimizers.RMSprop(learning_rate=0.005)
    model.compile(loss='mse', optimizer=opt, metrics=['mse']) # optimization_list[0]

    model.fit(Xx_train, yy_train, epochs=20)
    # model.fit(X_train, y_train, epochs=10)
    y_pred_nn = model.predict(Xx_test)
    sqrd_err_2 = metrics.mean_squared_error(yy_test, y_pred_nn)
    print('MSE with neural network:', sqrd_err_2)
    print_my_score(sqrd_err_2)

    # Neural Network Output
    output_predicted_2 = model.predict(X_output)
    print("Output 2: ", output_predicted_2.flatten())
    print(len(output_predicted_2), print(output_predicted_2[0]))
    create_csv_file(y_ids, output_predicted_2.flatten(), 'neural_network_output')


# Using Ridge Regression
def training_ridge_regression(Xx_train, yy_train, Xx_test, yy_test):
    # Training and testing a linear regression model
    ridge = Ridge().fit(Xx_train, yy_train)

    y_pred = ridge.predict(Xx_test)
    sqrd_err = metrics.mean_squared_error(yy_test, y_pred)
    print('\nMSE with Ridge regression:', sqrd_err)
    print_my_score(sqrd_err)


# Array of columns that need to be factorized
train_factorize = ['Gender',
                   'Relationship_Status',
                   'Hometown',
                   'Unit',
                   'Decision_skill_possess',
                   'Compensation_and_Benefits']
test_factorize = ['Gender',
                  'Relationship_Status',
                  'Hometown', 'Unit',
                  'Decision_skill_possess',
                  'Compensation_and_Benefits']
train_actual_columns = ['Employee_ID', 'Gender', 'Age', 'Education_Level',
                        'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess',
                        'Time_of_service', 'Time_since_promotion', 'growth_rate', 'Travel_Rate',
                        'Post_Level', 'Pay_Scale', 'Compensation_and_Benefits',
                        'Work_Life_balance', 'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6',
                        'VAR7', 'Attrition_rate']
test_actual_columns = ['Employee_ID', 'Gender', 'Age', 'Education_Level',
                       'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess',
                       'Time_of_service', 'Time_since_promotion', 'growth_rate', 'Travel_Rate',
                       'Post_Level', 'Pay_Scale', 'Compensation_and_Benefits',
                       'Work_Life_balance', 'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6',
                       'VAR7']
columns_na = ['Gender', 'Age', 'Education_Level', 'Relationship_Status', 'Hometown',
              'Unit', 'Decision_skill_possess', 'Time_of_service', 'Time_since_promotion',
              'growth_rate', 'Travel_Rate', 'Post_Level', 'Pay_Scale', 'Compensation_and_Benefits',
              'Work_Life_balance', 'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6', 'VAR7']


# reading csv files into data frames
train_df = pd.read_csv('./Data/Train.csv')
test_df = pd.read_csv('./Data/Test.csv')


# Factorize the data frames...
train_df = factorize_df(train_df, train_factorize)
test_df = factorize_df(test_df, test_factorize)


# Filling null values with the means
train_df = filling_na(train_df, columns_na)
test_df = filling_na(test_df, columns_na)


# # Checking the dataset shape
# print(train_df.shape)
# print(test_df.shape)
# # Getting the columns that need to be factorized
# print(train_df.columns)
# print(test_df.columns)


# Loading the data
train_df = train_df.to_numpy()
X = train_df[1:7000, 1:22]
y = train_df[1:7000, 23]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)


# Setting up the .csv files output
test_df = test_df.to_numpy()
X_test_output = test_df[0:3002, 1:22]
# print(test_df[0, 1])
y_ids = test_df[0:3002, 0]


training_linear_regression(X_train, y_train, X_test, y_test, X_test_output, y_ids)
training_neural_network(X_train, y_train, X_test, y_test, X_test_output, y_ids)
training_ridge_regression(X_train, y_train, X_test, y_test)


# View statistical details of the data
# print(train_df.head())
# print(test_df.head())




