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
steps
1. set all vars and load data/dbs/etc
# the needed external data should be pulled from its own pre class
# and fed here then it uses the arrange data class
2. check needed columns that have to be added changed
3. run regressions based on given 
"""

# 188, start there
class MachinePredictModel:

	# columns_all should have target name in it, both columns all and target
	# should just be the names of columns, with all being an array
	# target is target column to change
	def __init__(self, dataframe, columns_all, random_state, **kwargs):
		self.dataframe =  dataframe
		# columns all should contain all features that will be used plus the one target
		self.columns_all = columns_all
		# variable to be turned ino binary, class measure
		#self.target = target
		self.random_state = random_state
		# target_col_name is the create column that models will predict on
		self.date_unix = date_unix = kwargs.get('date_unix', None)
		self.time_interval_check = kwargs.get('time_interval_check', None)
		self.normalize_columns_array = kwargs.get('normalize_columns_array', None)
		self.time_period_returns_dict = kwargs.get('time_period_returns_dict', None)
		self.cols_to_drop = kwargs.get('cols_to_drop', None)
		self.target = kwargs.get('target', None)
		self.target_change_bin_dict = kwargs.get('target_change_bin_dict', None)
		self.col_to_make_target = kwargs.get('col_to_make_target', None)
		self.target_col_name = kwargs.get('target_col_name', None)
		self.target_amount = kwargs.get('target_amount', None)

	# this method is an interal class method to clean up date
	# what still needs to be added
	# 1. way to change needed columns data types such as turn all numerical
	# 2. set be able to set multi class vars like time of day - evebing, night etc
	# 3. drop columns if certain percent data is missing
	def _set_up_data_for_prob_predict(self, **kwargs):
		# **kawrgs vars below
		# initiate the data class
		model_dataframe = ArrangeData(self.dataframe)
		#print(time_interval_check, date_unix)
		# check if date_unix = none
		# if not none creates timedate
		if self.date_unix != None:
			model_dataframe.format_unix_date(self.date_unix)
		# this will eventually take in a dictionary  but first
		# the arrange data resample_date needs to be refactored for version .2 change	
		if self.time_interval_check == 1:
			model_dataframe.resample_date(self.target, 'month_highs_avg', 'M', 'mean')
			model_dataframe.resample_date(self.target, 'week_highs_avg', 'W', 'mean')
			model_dataframe.resample_date(self.target, 'day_highs_avg', 'D', 'mean')
		# normalize the given columns, with a new name which is always orginal 
		# column name + normalized
		if self.normalize_columns_array is not None:
			model_dataframe.normalize_new_column(self.normalize_columns_array)
		# takes in a dict, always has the same keys, column_name_old, column_name_new,
		# freq and returns new columns based on name of given time period return
		if self.time_period_returns_dict is not None:
			model_dataframe.time_period_returns_dict(self.time_period_returns_dict)
		if self.cols_to_drop is not None:
			model_dataframe.drop_columns(self.cols_to_drop)
		if self.target_change_bin_dict is not None:
			#model_dataframe.set_binary(self.col_to_make_target, self.target_col_name, self.target_amount)
			model_dataframe.set_binary_from_dict(self.target_change_bin_dict)
		model_dataframe.overall_data_display(8)
		return model_dataframe
		# everything above is setting up data, more still needs to be added
		# now comes the regressions on the bottom
		# there should be some type of dict model takes in with which models to run 
		# and which variables/error metrics to use etc
		# in fact the above method may become only class method
		# actually lets do that

	def predict_prob_model(self, training_percent, kfold_number, target_col_name, **kwargs):
		df = self._set_up_data_for_prob_predict()
		param_dict_logistic = kwargs.get('param_dict_logistic', None)
		param_dict_decision_tree = kwargs.get('param_dict_decision_tree', None)
		# set up features and target
		df.shuffle_rows()
		x_y_vars = df.set_features_and_target1(self.columns_all, target_col_name)
		features = x_y_vars[0]
		target = x_y_vars[1]
		# set up training and testing data
		vars_for_train_test = df.create_train_and_test_data_x_y_mixer(training_percent, features,target)
		X_train = vars_for_train_test[0]
		y_train = vars_for_train_test[1]
		X_test = vars_for_train_test[2]
		y_test = vars_for_train_test[3]

		# 1st model test logistic regression
		regres_instance = Regression(features, target, self.random_state)
		if param_dict_logistic is None:
			print(' didnt pick up first kwarg')
			log_regress_data = regres_instance.logistic_regres_with_kfold_cross_val()
		else:
			print('picked up params')
			log_regress_data = regres_instance.logistic_regres_with_kfold_cross_val(param_dict_logistic=param_dict_logistic)
		print(log_regress_data)
		if param_dict_decision_tree is None:
			print('no vars picked up')
		else:
			print(param_dict_decision_tree)
		# 2nd model test decision tree		


"""
# info for btc_data
file_location_btc = '/home/mike/Documents/coding_all/machine_predict/btc_play_data.csv'
file_location_loans = '/home/mike/Documents/coding_all/machine_predict/cleaned_loans_2007.csv'
df = pd.read_csv(file_location_btc)
columns = ['EUR_BTC_EX_High', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
columns_all = ['target_new', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
# these columns may or may not be created but target needs to be in col list
target = 'USD_BTC_EX_High'
normalize_columns_array = ['Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
random_state = 1
# method vars
#target_amount =.05
#target_col_name = 'target_new'
#col_to_make_target = 'week_highs_avg_change'
create_target_dict = {'column_name_old':['week_highs_avg_change','3day_highs_avg_change'], 'column_name_new':['target_new', '3day_highs_avg_change_bin_value'], 'value':[.05, .01]}
columns_to_drop = []
training_percent =.08
kfold_number = 10
target_col = create_target_dict['column_name_new'][0]
#**kwargs
kwarg_dict = {'time_interval_check':1, 'date_unix':'date_unix'}
time_interval_check = 1
date_unix = 'date_unix'
time_period_returns_dict = {'column_name_old':['week_highs_avg', 'day_highs_avg'], 'column_name_new':['week_highs_avg_change', '3day_highs_avg_change'], 'freq':[1,3]}
logistic_regression_params = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':'balanced', 'random_state':random_state, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}

# target_amount=target_amount, target_col_name=target_col_name, col_to_make_target=col_to_make_target,
# error
predict = MachinePredictModel(df, columns_all, random_state,  target=target, time_interval_check=1, date_unix='date_unix', normalize_columns_array=normalize_columns_array, time_period_returns_dict=time_period_returns_dict, target_change_bin_dict=create_target_dict)
df_rdy = predict._set_up_data_for_prob_predict()
print(type(df_rdy))
df_rdy.overall_data_display(10)
predict.predict_prob_model(training_percent, kfold_number, target_col, param_dict=logistic_regression_params)
# btc end 
"""


#info for loans
lend_tree_loan_data = '/home/mike/Documents/coding_all/data_sets_machine_predict/cleaned_loans_2007.csv'
df_loans = pd.read_csv(lend_tree_loan_data)
columns_all_loans = ['loan_amnt', 'int_rate', 'installment', 'emp_length', 'annual_inc',
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
target_loan = 'loan_status' 
target_loan = 'loan_status'
random_state_loan = 1
training_percent_loan = .08
kfold_number_loan = 10 
logistic_regression_params_loan = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':'balanced', 'random_state':random_state_loan, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
# max depht and min samples leaf can clash 
#decision_tree_params_loan = [criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False]
decision_tree_params_loan = ['test']
loan_predict = MachinePredictModel(df_loans, columns_all_loans, random_state_loan)
loan_predict._set_up_data_for_prob_predict()
loan_predict.predict_prob_model(training_percent_loan, kfold_number_loan, target_loan, param_dict_logistic=logistic_regression_params_loan, param_dict_decision_tree=decision_tree_params_loan)