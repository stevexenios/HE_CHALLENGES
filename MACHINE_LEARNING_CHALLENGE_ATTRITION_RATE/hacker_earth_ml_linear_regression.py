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
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from scipy.interpolate import PchipInterpolator, pchip_interpolate


# Displaying all columns
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
        the_df[i] = the_df[i].fillna((the_df[i].mean()))
    return the_df


# Function to calculate my score from SQRD
def print_my_score(sq_value):
    score = 100 * (1-sq_value)
    print("The score is: ", score)


# Function to create a csv file from data
def create_csv_file(y_id, y_predi, name):
    # Initialize data of lists
    result = {'Employee_ID': y_id, 'Attrition_rate': y_predi}
    result_df = pd.DataFrame(result)

    # Drop nameless column
    # result_df.drop(result_df.columns[result_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    result_df.to_csv(name, index=False)


# Plot Historgram for each column so as to help see which need Normalization
def plot_histogram(df, name_str):
    print(name_str + ' Plotting: ')
    df.hist()
    plt.show()


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


# Function to normalize the data in the columns
def normalize_attrition_rate(list_values):
    list_values = (list_values**2)**0.5
    print("Max: ", list_values.max(axis=0))
    print("Min: ", list_values.min(axis=0))
    print("Mean: ", list_values.mean(axis=0))

    summ = np.sum(list_values, axis=0, dtype=np.float)
    return list_values * list_values.mean(axis=0) / summ


# Feature engineering
def feature_engineering_columns(df):
    new_cols = ['VAR11', 'VAR22']
    df[new_cols[0]] = ((df['VAR1'] + df['VAR4'] + df['VAR5'] + df['VAR6'] - df['VAR2'] + df['VAR3'])**2)**0.5
    df[new_cols[1]] = df['Time_of_service'] - df['Time_since_promotion']

    return df


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

# Do some feature engineering
train_df = feature_engineering_columns(train_df)
test_df = feature_engineering_columns(test_df)
features_used = 2


# Plot to see if we need normalization
# plot_histogram(train_df, "Before")
#plot_histogram(test_df)


# Normalize the columns
# train_df = normalize_columns(train_df)
# plot_histogram(train_df, "After")

# # Checking the dataset shape
# print(train_df.shape)
# print(test_df.shape)
# # Getting the columns that need to be factorized
# print(train_df.columns)
# print(test_df.columns)


# Loading the data
train_df = train_df.to_numpy()
X = train_df[1:7000, 1:22+features_used]
y = train_df[1:7000, 23+features_used]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)


# Training and testing a linear regression model
linreg = LinearRegression()
linreg.fit(X_train, y_train)


y_pred = linreg.predict(X_test)
sqrd_err = metrics.mean_squared_error(y_test, y_pred)
print('\nMSE with linear regression:', sqrd_err)
print_my_score(sqrd_err)


test_df = test_df.to_numpy()
X_test = test_df[0:3002, 1:22+features_used]
# print(test_df[0, 1])
y_ids = test_df[0:3002, 0]
y_pred = linreg.predict(X_test)
y_pred = normalize_attrition_rate(y_pred)
create_csv_file(y_ids, y_pred, 'result_values.csv')

# View statistical details of the data
# print(train_df.head())
# print(test_df.head())

