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
		self.set_multi_class = kwargs.get('set_multi_class', None)
		self.convert_unix_to_human_date = kwargs.get('convert_unix_to_human_date', None)

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
		# this takes in an array, column name of date is first, then 1 to 
		# make new rows of the units separted by y,m,d,h,m,s,ms
		# array must have 2 variables 
		if self.convert_unix_to_human_date is not None:
			model_dataframe.convert_unix_to_human_date(self.convert_unix_to_human_date)	
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
		if self.set_multi_class is not None:
			model_dataframe.set_multi_class_array(self.set_multi_class)
		model_dataframe.overall_data_display(8)
		return model_dataframe
		# everything above is setting up data, more still needs to be added
		# now comes the regressions on the bottom
		# there should be some type of dict model takes in with which models to run 
		# and which variables/error metrics to use etc
		# in fact the above method may become only class method
		# actually lets do that

	def _get_error_scores_with_tpr_fpr(self, y_target, predictions):
		tp_filter = (predictions == 1) & (y_target == 1)
		tn_filter = (predictions == 0) & (y_target == 0)
		fp_filter = (predictions == 1) & (y_target == 0)
		fn_filter = (predictions == 0) & (y_target == 1)
		tp = len(predictions[tp_filter])
		tn = len(predictions[tn_filter])
		fp = len(predictions[fp_filter])
		fn = len(predictions[fn_filter])
		true_positive_rate = tp / (tp+fn)
		false_positive_rate = fp / (fp + tn)
		dict ={}
		y = y_target
		dict['roc_auc_score'] = roc_auc_score(y, predictions)
		dict['mse'] = mean_squared_error(y, predictions)
		dict['mae'] = mean_absolute_error(y, predictions)
		dict['r2_score'] = r2_score(y, predictions)
		dict['variance'] = np.var(predictions)
		dict['tpr'] = true_positive_rate
		dict['fpr'] = false_positive_rate
		return(dict)

	def _get_error_scores(self, y_target, predictions):
		y = y_target
		dict ={}
		dict['roc_auc_score'] = roc_auc_score(y, predictions)
		dict['mse'] = mean_squared_error(y, predictions)
		dict['mae'] = mean_absolute_error(y, predictions)
		dict['r2_score'] = r2_score(y, predictions)
		dict['variance'] = np.var(predictions)
		return dict
		

	def predict_prob_model(self, training_percent, kfold_number, target_col_name, **kwargs):
		df = self._set_up_data_for_prob_predict()
		param_dict_logistic = kwargs.get('param_dict_logistic', None)
		param_dict_decision_tree = kwargs.get('param_dict_decision_tree', None)
		param_dict_neural_network = kwargs.get('param_dict_neural_network', None)
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
		ppm_results_dict = {}
		# 1st model test logistic regression
		regres_instance = Regression(features, target, self.random_state)
		if param_dict_logistic is None:
			print(' didnt pick up first kwarg')
			ppm_results_dict['log_regress_data'] = regres_instance.logistic_regres_with_kfold_cross_val()
		else:
			print('picked up params')
			ppm_results_dict['log_regress_data'] = regres_instance.logistic_regres_with_kfold_cross_val(param_dict_logistic=param_dict_logistic)
		#print(log_regress_data)

		#2nd model test decision tree
		decision_tree_instance = DecisionTree('place_holder')
		if param_dict_decision_tree is None:
			print('no vars picked up')
			ppm_results_dict['decision_tree_data'] = decision_tree_instance.basic_tree_with_vars(X_train, y_train, X_test, y_test)
		else:
			print('vars passed to decision tree class')
			ppm_results_dict['decision_tree_data'] = decision_tree_instance.basic_tree_with_vars(X_train, y_train, X_test, y_test, param_dict_decision_tree=param_dict_decision_tree)
		#print(decision_tree_data)

		# 3rd model nueral network
		if param_dict_neural_network is None:
			ppm_results_dict['nnl_instance'] = NNet3(learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4)
		else:
			nnl_instance = NNet3(learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4, param_dict_neural_network=param_dict_neural_network)
			ppm_results_dict['nnl_data'] = nnl_instance.neural_learn_sk(X_train, y_train, X_test, y_test)
		#print(nnl_data)
		return ppm_results_dict

	def predict_prob_model_fit_parameters(self, training_percent, kfold_number, target_col_name, **kwargs):
		df = self._set_up_data_for_prob_predict()
		param_dict_logistic_array = kwargs.get('param_dict_logistic_array', None)
		param_dict_decision_tree_array = kwargs.get('param_dict_decision_tree_array', None)
		param_dict_neural_network_array = kwargs.get('param_dict_neural_network_array', None)
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
		dtree = DecisionTreeClassifier()
		reg = LogisticRegression()
		nnl = MLPClassifier()
		if param_dict_logistic_array is not None:
			clf = GridSearchCV(reg, param_dict_logistic_array)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_test)
			error_score = self._get_error_scores_with_tpr_fpr(y_test, predictions)
			error_score['best_Score'] = clf.best_score_
			error_score['best_params'] = clf.best_params_
			print(error_score)
		if param_dict_decision_tree_array is not None:
			clf = GridSearchCV(dtree, param_dict_decision_tree_array)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_test)
			error_score = self._get_error_scores_with_tpr_fpr(y_test, predictions)
			error_score['best_Score'] = clf.best_score_
			error_score['best_params'] = clf.best_params_
			print(error_score)
		if param_dict_neural_network_array is not None:
			clf = GridSearchCV(nnl, param_dict_neural_network_array)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_test)
			error_score = self._get_error_scores_with_tpr_fpr(y_test, predictions)
			error_score['best_Score'] = clf.best_score_
			error_score['best_params'] = clf.best_params_
			print(error_score)

	# columns all is an array 
	def cycle_vars(self, columns_all, training_percent, kfold_number, target_col_name):
		dict = {}
		for x in range(1, len(columns_all)+1):
			kicker = x
			start = 0
			end = start+ kicker
			cols = columns_all[start:end]
			data = self.predict_prob_model_fit_parameters(training_percent, kfold_number, target_col_name)
			x +=1
			dict[str(cols)] = dataframe
		return dict

	def cycle_vars_thru_features:(self, columns_all, target_col_name):
		max = len(columns_all)-



"""
# cycle vars example
	def cycle_vars_simple_lin_regress(self, columns, target):
		results_array = []
		dict = {}
		for x in range(1, len(columns)+1):
			kicker = x
			start = 0
			end = start + kicker
			cols = columns[start:end]
			instance = ArrangeData(df)
			add = instance.simple_lin_regres(columns, target)
			results_array.append(add)
			x += 1
			dict[str(cols)] = add
		return dict
"""



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

"""
#info for loans
lend_tree_loan_data = '/home/mike/Documents/coding_all/machine_predict/cleaned_loans_2007.csv'
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
decision_tree_params_loan = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':random_state_loan, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':'balanced', 'presort':False}
#decision_tree_params_loan = ['test']
nnl_params_loan = {'hidden_layer_sizes':(100, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':random_state_loan}
#param_dict_neural_network=nnl_params_loan
loan_predict = MachinePredictModel(df_loans, columns_all_loans, random_state_loan)
loan_predict._set_up_data_for_prob_predict()
loan_predict.predict_prob_model(training_percent_loan, kfold_number_loan, target_loan, param_dict_logistic=logistic_regression_params_loan, param_dict_decision_tree=decision_tree_params_loan,param_dict_neural_network=nnl_params_loan)
#loan_predict.predict_prob_model(training_percent_loan, kfold_number_loan, target_loan, param_dict_logistic=logistic_regression_params_loan)
"""

# info for bikes
file_location = '/home/mike/Documents/coding_all/machine_predict/hour.csv'
df_bike = pd.read_csv(file_location)
columns_to_drop_bike = ['casual', 'registered', 'dtedat']
columns_all_bike = ['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt_binary']
create_target_dict_bike = {'column_name_old':['cnt'], 'column_name_new':['cnt_binary'], 'value':[10]}
target_bike = 'cnt_binary'
set_multi_class_bike = ['hr', 6, 12, 18, 24 , 'hr_new']
random_state_bike = 1
training_percent_bike = .08
kfold_number_bike = 10 
logistic_regression_params_bike = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':'balanced', 'random_state':random_state_bike, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
# max depht and min samples leaf can clash 
decision_tree_params_bike = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':random_state_bike, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':'balanced', 'presort':False}
#decision_tree_params_loan = ['test']
nnl_params_bike = {'hidden_layer_sizes':(100, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':random_state_bike}
# bike model....
bike_predict = MachinePredictModel(df_bike, columns_all_bike, random_state_bike, cols_to_drop=columns_to_drop_bike,set_multi_class=set_multi_class_bike, target_change_bin_dict=create_target_dict_bike)
bike_predict._set_up_data_for_prob_predict()
results = bike_predict.predict_prob_model(training_percent_bike, kfold_number_bike, target_bike, param_dict_logistic=logistic_regression_params_bike, param_dict_decision_tree=decision_tree_params_bike,param_dict_neural_network=nnl_params_bike)
# bike model for optimizing 
# range of values in dict form for parameters
decision_tree_array_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':[None, 'balanced'], 'random_state':[random_state_bike]}
logistic_regression_array_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1], 'random_state':[random_state_bike]}
neural_net_array_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[True], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08], 'random_state':[random_state_bike]}
# optimize model 
#bike_predict.predict_prob_model_fit_parameters(training_percent_bike, kfold_number_bike, target_bike, param_dict_logistic_array=logistic_regression_array_vars, param_dict_decision_tree_array=decision_tree_array_vars, param_dict_neural_network_array=neural_net_array_vars)
#bike_predict.predict_prob_model_fit_parameters(training_percent_bike, kfold_number_bike, target_bike, param_dict_decision_tree_array=decision_tree_array_vars)
#print(results)
columns_all_features_bike = 
results2 = bike_predict.cycle_vars(columns_all_features_bike, training_percent_bike, kfold_number_bike, target_bike)

"""
#thoughts 
1. everything above is for classifers
2. could models be mixed and match for different decisions, such as decision tree to predict when right
and nnl to weight when wrong. 
3. need to set up to iterate over multi time periods on data and cycle thru vars 
"""

"""
how this should be
1. put in various vars on top
2. pick models to run
3. pick tpr and fpr ranges, with error compared against some
t value and return if statistically sifnifcant or not
4. print out data, which model, which params, which vars, and score (error, tpr, fpr)
5. run that on new data not seen (time period ahead)
"""

"""
whats next.....
in this order
1. refactor so everything is taken in at start of class
2. refactor all regressions to out fit, and out put regressions
3. have the error scores be part of this MachinePredictModel class and return everything as dict
4. set up method to iterate over various varaiables
5. check if chose error matetric is stat significant
6. if stat significant return return scores if they hit a certain range