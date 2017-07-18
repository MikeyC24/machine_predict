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
from sklearn.model_selection import ParameterGrid, GridSearchCV
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

"""
# this play uses btc_play data on all three models
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
"""

# below the focus is to run the regression model with different combo
# of variables and out their input as well as
# different combos for the regression class
# first on bike data and decision tree
random_state = 1
file_location = '/home/mike/Documents/coding_all/machine_predict/hour.csv'
df = pd.read_csv(file_location)
bikes = ArrangeData(df)
bikes.overall_data_display(1)
columns_to_drop = ['casual', 'registered', 'dtedat']
bikes.drop_columns(columns_to_drop)
columns_all = ['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt_binary']
bikes.set_binary('cnt', 'cnt_binary', 15)
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

# load data
file_location = '/home/mike/Documents/coding_all/machine_predict/hour.csv'
df = pd.read_csv(file_location)
# variables needed 
random_state = 1
train_percent = .08
# set up Arrange data class
bikes = ArrangeData(df)
bikes.overall_data_display(1)
all_columns = ['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt_binary', 'casual', 'registered', 'dtedat']
columns_to_drop = ['casual', 'registered', 'dtedat']
# need function to return an array of columns taking out thr dropped
# columns from all
columns_minus_dropped = ['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt_binary']
target = 'cnt_binary'
# drop columns not needed
bikes.drop_columns(columns_to_drop)
# set columns that need to be to classes
# how can this be automated? gotta be a function where you can 
# enter in number of classes wanted and any number of ranges
bikes.set_mutli_class('hr', 6, 12, 18, 24 , 'hr_new')
# set target column as binary
bikes.set_binary('cnt', 'cnt_binary', 15)
# df instance as var
df_instance = bikes
# set features and target
x_y_vars = bikes.set_features_and_target1(columns_minus_dropped, target)
features = x_y_vars[0]
target = x_y_vars[1]
# set train and test
varsxy = bikes.create_train_and_test_data_x_y_mixer(train_percent,features,target)
X_train = varsxy[0]
y_train = varsxy[1]
X_test = varsxy[2]
y_test = varsxy[3]
# will only focus on basic_tree_with_vars
tree_vars = {'min_samples_split':[3,50,100], 'max_depth':[2,10]}
tree_vars1 = dict = {'min_samples_split':[4,50,100]}
#print(len(tree_vars1))
# set up tree class
tree_instance = DecisionTree(random_state)
# iteration to get all needed variables from a dict for the class instances
# now need to have it set up where it runs thru every combo possible


for key,values in tree_vars.items():
	for x in range(len(values)):
		var=values[x]
		print(var)
		#print(key +'='+ values[x])
		#tree_results = tree_instance.basic_tree_with_vars(X_train, y_train, X_test, y_test, min_samples_split=var, max_depth=5)
		#print(tree_results)
tree_vars_full = {'X_train':[X_train], 'y_train':[y_train], 'X_test':[X_test], 'y_test':[y_test],'min_samples_split':[3,50,100], 'max_depth':[2,10]}
# this is no quite all the vars
decision_tree_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':[None, 'balanced']}
logistic_regression_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1]}
neural_net_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50,200], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[False], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08]}
iterate_params = list(ParameterGrid(decision_tree_vars))
#print(iterate_params)
print(len(iterate_params))
for x in range(len(iterate_params)):
	var = iterate_params[x]
	#print(var)
	#tree_results = tree_instance.basic_tree_with_vars(X_train, y_train, X_test, y_test, var)
	#print(tree_results)
"""
for key,values in tree_vars_full.items():
	for x in range(len(values)):
		var=values[x]
		print(var)
		#print(key +'='+ values[x])
		#tree_results = tree_instance.basic_tree_with_vars(X_train, y_train, X_test, y_test, min_samples_split=var, max_depth=5)
		#print(tree_results)
"""
"""
# GridSearchCV
# working for all three, try on btc_data?
dtree = DecisionTreeClassifier(random_state)
reg = LogisticRegression(random_state)
nnl = MLPClassifier(random_state)
dtree_parameters = decision_tree_vars
reg_parameters = logistic_regression_vars
nnl_params = neural_net_vars
#clf = GridSearchCV(dtree, dtree_parameters)
clf = GridSearchCV(reg, reg_parameters)
clf.fit(X_train, y_train)
#get_params = clf.get_params
#print(get_params)
predictions = clf.predict(X_test)
auc = roc_auc_score(y_test, predictions)
print(auc)
print('Best score: {0}'.format(clf.best_score_))  
print(clf.best_params_) 

print('_________________________')

# , n_jobs=2, cv=5, verbose=2, pre_dispatch='2*n_jobs', refit=True)
dtree = DecisionTreeClassifier()
parameters = decision_tree_vars
data_Set = df_instance
clf1 = GridSearchCV(dtree, parameters,n_jobs=2, cv=5, pre_dispatch='2*n_jobs', refit=True)
clf1.fit(X_train, y_train)
#get_params = clf1.get_params
#print(get_params)
predictions = clf1.predict(X_test)
auc1 = roc_auc_score(y_test, predictions)
print(auc)
print('Best score: {0}'.format(clf1.best_score_))  
print(clf1.best_params_) 

print('_________________________')

clf2 = GridSearchCV(nnl, nnl_params)
clf2.fit(X_train, y_train)
#get_params = clf.get_params
#print(get_params)
predictions = clf2.predict(X_test)
auc2 = roc_auc_score(y_test, predictions)
print(auc)
print('Best score: {0}'.format(clf2.best_score_))  
print(clf2.best_params_) 
"""

"""
# this play uses btc_play data on all three models
# but for grid search
file_location = '/home/mike/Documents/coding_all/machine_predict/btc_play_data.csv'
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
btc_play_data.set_binary('3days_highs_avg_change', 'target_new', '.1')
btc_play_data.set_mutli_class('3days_highs_avg_change', -.03, 0, .02, .04, 'class_form_3d_high_chage')
btc_play_data.set_ones()
columns_all = ['target_new', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
target = 'target_new'
btc_play_data.shuffle_rows()
x_y_vars = btc_play_data.set_features_and_target1(columns_all, target)
features = x_y_vars[0]
target = x_y_vars[1]
cols_norm = ['Transactions_Volume_normalized', 'Number_of_Transactions_normalized', 'LTC_BTC_EX_High_normalized', 'EUR_BTC_EX_High_normalized']
cols =['EUR_BTC_EX_High', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
btc_play_data.normalize_new_column(cols_norm)
varsxy = btc_play_data.create_train_and_test_data_x_y_mixer(.8,features,target)
X_train = varsxy[0]
y_train = varsxy[1]
X_test = varsxy[2]
y_test = varsxy[3]
print(btc_play_data.overall_data_display(5))

# GridSearchCV
# for btc_data
decision_tree_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':[None, 'balanced']}
logistic_regression_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1]}
neural_net_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50,200], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[False], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08]}
dtree = DecisionTreeClassifier(random_state)
reg = LogisticRegression(random_state)
nnl = MLPClassifier(random_state)
dtree_parameters = decision_tree_vars
reg_parameters = logistic_regression_vars
nnl_params = neural_net_vars
#clf = GridSearchCV(dtree, dtree_parameters)
clf = GridSearchCV(reg, reg_parameters)
clf.fit(X_train, y_train)
#get_params = clf.get_params
#print(get_params)
predictions = clf.predict(X_test)
auc = roc_auc_score(y_test, predictions)
print(auc)
print(clf.best_estimator_)
#print(clf.cv_results_)
reg_best_score = clf.best_score_
reg_best_params = clf.best_params_
print('Best score: {0}'.format(clf.best_score_))  
print(clf.best_params_) 

print('_________________________')

parameters = decision_tree_vars
data_Set = df_instance
clf1 = GridSearchCV(dtree, parameters,n_jobs=2, cv=5, pre_dispatch='2*n_jobs', refit=True)
clf1.fit(X_train, y_train)
#get_params = clf1.get_params
#print(get_params)
predictions = clf1.predict(X_test)
auc1 = roc_auc_score(y_test, predictions)
print(auc)
print(clf1.best_estimator_)
#print(clf1.cv_results_)
tree_best_score = clf1.best_score_
tree_best_params = clf1.best_params_
print('Best score: {0}'.format(clf1.best_score_))  
print(clf1.best_params_) 

print('_________________________')

clf2 = GridSearchCV(nnl, nnl_params)
clf2.fit(X_train, y_train)
#get_params = clf.get_params
#print(get_params)
predictions = clf2.predict(X_test)
auc2 = roc_auc_score(y_test, predictions)
print(auc)
print(clf2.best_estimator_)
#print(clf2.cv_results_)
nnl_best_score = clf2.best_score_
nnl_best_params = clf2.best_params_
print('Best score: {0}'.format(clf2.best_score_))  
print(clf2.best_params_) 

dict_btc_vars = {'reg_best_score':reg_best_score,
'reg_best_params':reg_best_params,
'tree_best_score':tree_best_score,
'tree_best_params':tree_best_params,
'nnl_best_score':nnl_best_score,
'nnl_best_params':nnl_best_params }
print(dict_btc_vars)
"""

lend_tree_loan_data = '/home/mike/Documents/coding_all/machine_predict/cleaned_loans_2007.csv'
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
varsxy = loans_data.create_train_and_test_data_x_y_mixer(.8,features,target)
X_train = varsxy[0]
y_train = varsxy[1]
X_test = varsxy[2]
y_test = varsxy[3]
decision_tree_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':[None, 'balanced']}
logistic_regression_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1]}
neural_net_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50,200], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[True], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08]}
dtree = DecisionTreeClassifier(random_state)
reg = LogisticRegression(random_state)
nnl = MLPClassifier(random_state)
dtree_parameters = decision_tree_vars
reg_parameters = logistic_regression_vars
nnl_params = neural_net_vars
#clf = GridSearchCV(dtree, dtree_parameters)
clf = GridSearchCV(reg, reg_parameters)
clf.fit(X_train, y_train)
#get_params = clf.get_params
#print(get_params)
predictions = clf.predict(X_test)
auc = roc_auc_score(y_test, predictions)
print(auc)
print(clf.best_estimator_)
#print(clf.cv_results_)
reg_best_score = clf.best_score_
reg_best_params = clf.best_params_
print('Best score: {0}'.format(clf.best_score_))  
print(clf.best_params_) 

print('_________________________')

parameters = decision_tree_vars
data_Set = df_instance
clf1 = GridSearchCV(dtree, parameters,n_jobs=2, cv=5, pre_dispatch='2*n_jobs', refit=True)
clf1.fit(X_train, y_train)
#get_params = clf1.get_params
#print(get_params)
predictions = clf1.predict(X_test)
auc1 = roc_auc_score(y_test, predictions)
print(auc)
print(clf1.best_estimator_)
#print(clf1.cv_results_)
tree_best_score = clf1.best_score_
tree_best_params = clf1.best_params_
print('Best score: {0}'.format(clf1.best_score_))  
print(clf1.best_params_) 

print('_________________________')

clf2 = GridSearchCV(nnl, nnl_params)
clf2.fit(X_train, y_train)
#get_params = clf.get_params
#print(get_params)
predictions = clf2.predict(X_test)
auc2 = roc_auc_score(y_test, predictions)
print(auc)
print(clf2.best_estimator_)
#print(clf2.cv_results_)
nnl_best_score = clf2.best_score_
nnl_best_params = clf2.best_params_
print('Best score: {0}'.format(clf2.best_score_))  
print(clf2.best_params_) 

dict_loans_vars = {'reg_best_score':reg_best_score,
'reg_best_params':reg_best_params,
'tree_best_score':tree_best_score,
'tree_best_params':tree_best_params,
'nnl_best_score':nnl_best_score,
'nnl_best_params':nnl_best_params }
print(dict_btc_vars)