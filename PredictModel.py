import pandas as pd
import numpy as np
import datetime
import time
from itertools import cycle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc,log_loss, precision_score
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
import operator
from sklearn import preprocessing
# read this of above http://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import math
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from Regression import *
from ArrangeData import *
from DecisionTrees import *
from NeuralNetwork import *

"""
lend_tree_loan_data = '/home/mike/Documents/coding_all/data_sets_machine_predict/cleaned_loans_2007.csv'
df_loans = pd.read_csv(lend_tree_loan_data)
columns = ['loan_amnt', 'int_rate', 'installment', 'emp_length', 'annual_inc',
        'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
       'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
       'home_ownership_MORTGAGE', 'home_ownership_NONE',
       'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
       'verification_status_Not Verified',
       'verification_status_Source Verified', 'verification_status_Verified',
       'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation',
       'purpose_educational', 'purpose_home_improvement', 'purpose_house',
       'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
       'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
       'purpose_vacation', 'purpose_wedding', 'term_ 36 months',
       'term_ 60 months', 'loan_status']
target = 'loan_status'
loans_data = ArrangeData(df_loans)
#loans_data.overall_data_display(1)
random_state = 1
#print(type(df_loans))
#loans_data.show_dtypes('object')
#loans_data.show_dtypes('float')
x_y_vars = loans_data.set_features_and_target1(columns, target)
features = x_y_vars[0]
target = x_y_vars[1]
print(type(features))
print(type(target))
regres_instance = Regression(features, target, random_state)
print(type(regres_instance.features))
print(type(regres_instance.target))
data = regres_instance.logistic_regres_with_kfold_cross_val()
print(data)
"""
"""
file_location = '/home/mike/Documents/coding_all/data_sets_machine_predict/btc_play_data.csv'
df = pd.read_csv(file_location)
columns = ['EUR_BTC_EX_High', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
columns_all = ['target_new', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']

fold = 10
random_state = 1
btc_play_data = ArrangeData(df)
btc_play_data.format_unix_date('date_unix')
btc_play_data.normalize_new_column(columns)
btc_play_data.resample_date('USD_BTC_EX_High', 'month_highs_avg', 'M', 'mean')
btc_play_data.resample_date('USD_BTC_EX_High', 'week_highs_avg', 'W', 'mean')
btc_play_data.resample_date('USD_BTC_EX_High', 'day_highs_avg', 'D', 'mean')
btc_play_data.time_period_returns('week_highs_avg', 'week_highs_avg_change', freq=1)
btc_play_data.time_period_returns('day_highs_avg', '3days_highs_avg_change', freq=3)
btc_play_data.set_binary('week_highs_avg_change', 'target_new', '.05')
btc_play_data.set_binary('3days_highs_avg_change', 'target_new', '.05')
btc_play_data.set_mutli_class('3days_highs_avg_change', -.03, 0, .02, .04)
columns_all = ['target_new', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
target = 'target_new'
btc_play_data.overall_data_display(10)


x_y_vars = btc_play_data.set_features_and_target1(columns_all, target)
features = x_y_vars[0]
target = x_y_vars[1]
regres_instance = Regression(features, target, random_state)
data = regres_instance.logistic_regres_with_kfold_cross_val()
print(data)

"""

"""
below is bike data for decision treeclassifer and two neural nets
random_state = 1
file_location = '/home/mike/Documents/coding_all/machine_predict/hour.csv'
df = pd.read_csv(file_location)
bikes = ArrangeData(df)
bikes.overall_data_display(1)
columns_to_drop = ['casual', 'registered', 'dtedat']
bikes.drop_columns(columns_to_drop)
columns_all = ['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt_binary']
bikes.set_binary('cnt', 'cnt_binary', 10)
target = 'cnt_binary'
bikes.set_mutli_class('hr', 6, 12, 18, 24 , 'hr_new')
bikes.set_ones()
#bikes.overall_data_display(35)
dataframe = bikes
x_y_vars = bikes.set_features_and_target1(columns_all, target)
features = x_y_vars[0]
target = x_y_vars[1]
varsxy = bikes.create_train_and_test_data_x_y_mixer(.8,features,target)
X_train = varsxy[0]
y_train = varsxy[1]
X_test = varsxy[2]
y_test = varsxy[3]
tree_instance = DecisionTree(random_state)
basic_tree = tree_instance.basic_tree(X_train, y_train, X_test, y_test)
tree2 = tree_instance.basic_tree_with_vars(X_train, y_train, X_test, y_test, min_samples_split=5)
tree3 = tree_instance.random_forest_with_vars(X_train, y_train, X_test, y_test,min_samples_leaf=3,n_estimators=100)
print('_________________')
print(basic_tree)
print('_________________')
print(tree2)
print('_________________')
print(tree3)
# this is neural net predictions based on sklearn
nnet = NNet3(random_state=random_state)
net_predict = nnet.neural_learn_sk(X_train, y_train, X_test, y_test)
print(net_predict)

# below is neural net predictions based on dataquest model
# neither seems to do much 
bikes1 = ArrangeData(df)
#bikes1.overall_data_display(1)
columns_to_drop = ['casual', 'registered', 'dtedat']
bikes1.drop_columns(columns_to_drop)
columns_all = ['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt_binary']
bikes1.set_binary('cnt', 'cnt_binary', 10)
features = ['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
bikes1.set_binary('cnt', 'cnt_binary', 10)
target = 'cnt_binary'
bikes1.set_mutli_class('hr', 6, 12, 18, 24 , 'hr_new')
bikes1.set_ones()
# normalizing did not change anything 
#normal_cols = ['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
#bikes1.normalize_new_column(normal_cols)
bikes1.shuffle_rows()
df_bike = bikes1
x_y_vars = bikes1.set_x_y_vars_from_df(features, target)
X = x_y_vars[0]
y = x_y_vars[1]
#print(X)
X1 = df_bike.dataframe[['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']].values
y1 = df_bike.dataframe['cnt_binary'].values
#print(X1)
testing_vars  =bikes1.create_train_and_test_data_x_y_mixer(.08, X, y)

X_train1 = testing_vars[0] 
y_train1 = testing_vars[1]
X_test1 = testing_vars[2] 
y_test1 = testing_vars[3]
learning_rate = 0.5
# Maximum number of iterations for gradient descent
maxepochs = 10000       
# Costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.00001  
# Number of hidden units
hidden_units = 4
model1 = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
			  convergence_thres=convergence_thres, hidden_layer=hidden_units)
model1.learn(X_train1, y_train1)
predictions1 = model1.predict(X_test1)[0]
auc1 = roc_auc_score(y_test1, predictions1)
mse1 = mean_squared_error(y_test1, predictions1)
log_loss_var1 = log_loss(y_test1, predictions1)
#precision_score and accuracy cant handle mix of binary and contineous
#precision_score_var  = precision_score(y_test, predictions)
roccurve = fpr, tpr, thresholds = roc_curve(y_test1, predictions1)
print(auc1, log_loss_var1)
"""


file_location = '/home/mike/Documents/coding_all/data_sets_machine_predict/btc_play_data.csv'
df = pd.read_csv(file_location)
columns = ['EUR_BTC_EX_High', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
columns_all = ['target_new', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']

fold = 10
random_state = 1
btc_play_data = ArrangeData(df)
btc_play_data.format_unix_date('date_unix')
btc_play_data.normalize_new_column(columns)
btc_play_data.resample_date('USD_BTC_EX_High', 'month_highs_avg', 'M', 'mean')
btc_play_data.resample_date('USD_BTC_EX_High', 'week_highs_avg', 'W', 'mean')
btc_play_data.resample_date('USD_BTC_EX_High', 'day_highs_avg', 'D', 'mean')
btc_play_data.time_period_returns('week_highs_avg', 'week_highs_avg_change', freq=1)
btc_play_data.time_period_returns('day_highs_avg', '3days_highs_avg_change', freq=3)
btc_play_data.set_binary('week_highs_avg_change', 'target_new', '.01')
btc_play_data.set_binary('3days_highs_avg_change', 'target_new', '.05')
btc_play_data.set_mutli_class('3days_highs_avg_change', -.03, 0, .02, .04, 'class_form_3d_high_chage')
btc_play_data.set_ones()
columns_all = ['target_new', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
target = 'target_new'
btc_play_data.shuffle_rows()
btc_play_data.overall_data_display(10)

# logsitic regression
x_y_vars = btc_play_data.set_features_and_target1(columns_all, target)
features = x_y_vars[0]
target = x_y_vars[1]
regres_instance = Regression(features, target, random_state)
data = regres_instance.logistic_regres_with_kfold_cross_val()
print(data)

cols_norm = ['ones', 'Transactions_Volume_normalized', 'Number_of_Transactions_normalized', 'LTC_BTC_EX_High_normalized', 'EUR_BTC_EX_High_normalized']
cols =['EUR_BTC_EX_High', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
btc_play_data.normalize_new_column(cols_norm)
varsxy = btc_play_data.create_train_and_test_data_x_y_mixer(.8,features,target)
X_train = varsxy[0]
y_train = varsxy[1]
X_test = varsxy[2]
y_test = varsxy[3]
tree_instance = DecisionTree(random_state)
basic_tree = tree_instance.basic_tree(X_train, y_train, X_test, y_test)
tree2 = tree_instance.basic_tree_with_vars(X_train, y_train, X_test, y_test, min_samples_split=5)
tree3 = tree_instance.random_forest_with_vars(X_train, y_train, X_test, y_test,min_samples_leaf=3,n_estimators=100)
print('_________________')
print(basic_tree)
print('_________________')
print(tree2)
print('_________________')
print(tree3)
nnet = NNet3(random_state=random_state)
net_predict = nnet.neural_learn_sk(X_train, y_train, X_test, y_test)
print(net_predict)

features1_norm = ['ones', 'Transactions_Volume_normalized', 'Number_of_Transactions_normalized', 'LTC_BTC_EX_High_normalized', 'EUR_BTC_EX_High_normalized']
features1 = ['EUR_BTC_EX_High', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
target1 ='target_new'

x_y_vars = btc_play_data.set_x_y_vars_from_df(features1_norm, target1)
X = x_y_vars[0]
y = x_y_vars[1]
df_btc = btc_play_data
#print(X)
#X1 = df_btc.dataframe[['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']].values
#y1 = df_btc.dataframe['cnt_binary'].values
#print(X1)
testing_vars  =btc_play_data.create_train_and_test_data_x_y_mixer(.08, X, y)

X_train1 = testing_vars[0] 
y_train1 = testing_vars[1]
X_test1 = testing_vars[2] 
y_test1 = testing_vars[3]
learning_rate = 0.5
# Maximum number of iterations for gradient descent
maxepochs = 10000       
# Costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.00001  
# Number of hidden units
hidden_units = 4
model1 = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
			  convergence_thres=convergence_thres, hidden_layer=hidden_units)
model1.learn(X_train1, y_train1)
predictions1 = model1.predict(X_test1)[0]
auc1 = roc_auc_score(y_test1, predictions1)
mse1 = mean_squared_error(y_test1, predictions1)
log_loss_var1 = log_loss(y_test1, predictions1)
#precision_score and accuracy cant handle mix of binary and contineous
#precision_score_var  = precision_score(y_test, predictions)
roccurve = fpr, tpr, thresholds = roc_curve(y_test1, predictions1)
print(auc1, log_loss_var1)