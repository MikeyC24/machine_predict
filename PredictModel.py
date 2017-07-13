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
bikes.overall_data_display(35)
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
