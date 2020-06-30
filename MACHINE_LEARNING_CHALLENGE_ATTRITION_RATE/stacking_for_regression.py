# ML in Python, homework 3
# Date: 06/19/2019
# Author: Steve G. Mwangi
# Description: Neural network for predicting attrition rate of employees
from keras.engine.saving import load_model
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from sklearn.ensemble import StackingRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
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
        the_df[i] = the_df[i].fillna((the_df[i].mean()))
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


# Get a list of models to evaluate
def get_models():
    models = dict()
    models['knn'] = KNeighborsRegressor()
    models['cart'] = DecisionTreeRegressor()
    models['svm'] = SVR()
    models['stacking'] = get_stacking()
    return models


# Evaluate a given model using cross-validation
def evaluate_model(model):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('knn', KNeighborsRegressor()))
    level0.append(('cart', DecisionTreeRegressor()))
    level0.append(('lr', RidgeCV()))
    # level0.append(('svm', SVR()))
    # define meta learner model
    level1 = LinearRegression()
    # define the stacking ensemble
    model1 = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model1


# Ada Boost Ensemble
def ensemble_ada_boost(X, y, X_test1, y_test1):
    regr = AdaBoostRegressor(random_state=None, n_estimators=100, learning_rate=0.1, loss='linear')
    regr.fit(X, y)
    y_pred1 = regr.predict(X_test1)
    sqrd_err1 = metrics.mean_squared_error(y_test1, y_pred1)
    print('\nMSE with ADA Boost ensemble:', sqrd_err1)
    print_my_score(sqrd_err)


# Random Forest Ensemble
def ensemble_random_forest(X, y, X_test1, y_test1):
    regr = RandomForestRegressor(max_depth=5, random_state=0)
    regr.fit(X, y)
    y_pred1 = regr.predict(X_test1)
    sqrd_err1 = metrics.mean_squared_error(y_test1, y_pred1)
    print('\nMSE with Random Forest ensemble:', sqrd_err1)
    print_my_score(sqrd_err)


# Gradient Boosting Regressor
def gradient_boosting_regressor(X, y, X_test1, y_test1):
    regr = GradientBoostingRegressor(random_state=0)
    regr.fit(X, y)
    y_pred1 = regr.predict(X_test1)
    sqrd_err1 = metrics.mean_squared_error(y_test1, y_pred1)
    print('\nMSE with Gradient Boosting Regressor:', sqrd_err1)
    print_my_score(sqrd_err)


# Training and testing a neural network
def training_neural_network(Xx_train, yy_train, Xx_test, yy_test, y_ds, x_output_test):
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
    opt = keras.optimizers.RMSprop(learning_rate=0.001, decay=0.1)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    model.fit(Xx_train, yy_train, epochs=20, verbose=1)
    # model.fit(X_train, y_train, epochs=10)
    y_pred_nn = model.predict(Xx_test)
    sqrd_err_2 = metrics.mean_squared_error(yy_test, y_pred_nn)
    print('MSE with neural network:', sqrd_err_2)
    print_my_score(sqrd_err_2)

    output_predicted_2 = model.predict(x_output_test)
    create_csv_file(y_ds, output_predicted_2.flatten(), 'nn_output')


# Stacked NNs
def nn_for_MLP(Xx_train, yy_train):
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
    model.compile(loss='mse', optimizer=optimization_list[0], metrics=['mse'])

    model.fit(Xx_train, yy_train, epochs=100, verbose=0)
    return model


# Creating a Meta-Learner that loads models from file
def load_all_models(n_models):
    all_models = list()
    for ii in range(n_models):
        # define filename for this ensemble
        filename1 = 'Models/model_' + str(ii + 1) + '.h5'
        # load model from file
        model1 = load_model(filename1)
        # add to list of members
        all_models.append(model1)
        print('>loaded %s' % filename1)
    return all_models


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


# Plot to see if we need normalization
# plot_histogram(train_df, "Before")
#plot_histogram(test_df)


# Normalize the columns
train_df = normalize_columns(train_df)
# plot_histogram(train_df, "After")

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)


# Training and testing a linear regression model
linreg = LinearRegression()
linreg.fit(X_train, y_train)


y_pred = linreg.predict(X_test)
sqrd_err = metrics.mean_squared_error(y_test, y_pred)
print('\nMSE with linear regression:', sqrd_err)
print_my_score(sqrd_err)


# # get the models to evaluate
# models = get_models()
# # evaluate the models and store results
# results, names = list(), list()


# for name, model in models.items():
#     scores = evaluate_model(model)
#     results.append(scores)
#     names.append(name)
#     print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
#
#
# # plot model performance for comparison
# pyplot.boxplot(results, labels=names, showmeans=True)
# pyplot.show()

model = get_stacking()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
sqrd_err = metrics.mean_squared_error(y_test, y_pred)
print('\nMSE with stacking (4 Models):', sqrd_err)
print_my_score(sqrd_err)


# Stacking Output
# Setting up the .csv files output
test_df = test_df.to_numpy()
X_test_output = test_df[0:3002, 1:22]
# print(test_df[0, 1])
y_ids = test_df[0:3002, 0]
output_predicted_1 = model.predict(X_test_output)
create_csv_file(y_ids, output_predicted_1, 'model_stacking_output')


ensemble_ada_boost(X_train, y_train, X_test, y_test)
ensemble_random_forest(X_train, y_train, X_test, y_test)
gradient_boosting_regressor(X_train, y_train, X_test, y_test)
training_neural_network(X_train, y_train, X_test, y_test, test_df[0:3002, 0], test_df[0:3002, 1:22])


# # MLP fit and save models
# n_members = 5
# for i in range(n_members):
#     # fit model
#     model = nn_for_MLP(X_train, y_train)
#     # save model
#     filename = 'Models/model_' + str(i + 1) + '.h5'
#     model.save(filename)
#     print('>Saved %s' % filename)
#
#
# members = load_all_models(n_members)
# print('Loaded %d models' % len(members))


# test_df = test_df.to_numpy()
# X_test = test_df[0:3002, 1:22]
# # print(test_df[0, 1])
# y_ids = test_df[0:3002, 0]
# y_pred = linreg.predict(X_test)
# create_csv_file(y_ids, y_pred, 'result_values.csv')

# View statistical details of the data
# print(train_df.head())
# print(test_df.head())

