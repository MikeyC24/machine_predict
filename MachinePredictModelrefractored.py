import pandas as pd
import numpy as np
import datetime
import time
#from itertools import cycle
import itertools
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
from RegressionCombined import *

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
	def __init__(self, dataframe, columns_all, random_state, training_percent, kfold_number, target_col_name, **kwargs):
		self.dataframe =  dataframe
		# columns all should contain all features that will be used plus the one target
		self.columns_all = columns_all
		self.random_state = random_state
		self.training_percent = training_percent
		self.kfold_number = kfold_number
		self.target_col_name = target_col_name
		self.date_unix = date_unix = kwargs.get('date_unix', None)
		self.time_interval_check = kwargs.get('time_interval_check', None)
		self.normalize_columns_array = kwargs.get('normalize_columns_array', None)
		self.time_period_returns_dict = kwargs.get('time_period_returns_dict', None)
		self.cols_to_drop = kwargs.get('cols_to_drop', None)
		self.target = kwargs.get('target', None)
		self.target_change_bin_dict = kwargs.get('target_change_bin_dict', None)
		#self.col_to_make_target = kwargs.get('col_to_make_target', None)
		self.target_amount = kwargs.get('target_amount', None)
		self.set_multi_class = kwargs.get('set_multi_class', None)
		self.convert_unix_to_human_date = kwargs.get('convert_unix_to_human_date', None)
		self.kfold_dict = kwargs.get('kfold_dict', None)
		self.param_dict_logistic = kwargs.get('param_dict_logistic', None)
		self.param_dict_decision_tree = kwargs.get('param_dict_decision_tree', None)
		self.param_dict_neural_network = kwargs.get('param_dict_neural_network', None)
		self.param_dict_logistic_array = kwargs.get('param_dict_logistic_array', None)
		self.param_dict_decision_tree_array = kwargs.get('param_dict_decision_tree_array', None)
		self.param_dict_neural_network_array = kwargs.get('param_dict_neural_network_array', None)
		self.user_input_for_model_output = kwargs.get('user_input_for_model_output', None)

	# this method is an interal class method to clean up date
	# what still needs to be added
	# 1. way to change needed columns data types such as turn all numerical
	# 2. set be able to set multi class vars like time of day - evebing, night etc
	# 3. drop columns if certain percent data is missing
	def _set_up_data_for_prob_predict(self):
		# **kawrgs vars below
		# initiate the data class
		model_dataframe = ArrangeData(self.dataframe)
		#print(time_interval_check, date_unix)
		# check if date_unix = none
		# if not none creates timedate
		if self.date_unix is not None:
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
		#model_dataframe.overall_data_display(8)
		return model_dataframe
		# everything above is setting up data, more still needs to be added
		# now comes the regressions on the bottom
		# there should be some type of dict model takes in with which models to run 
		# and which variables/error metrics to use etc
		# in fact the above method may become only class method
		# actually lets do that

	# take in df of cleaned model, return features, target, train and test data in dict
	def _set_up_data_for_models(self):
		model_dataframe = self._set_up_data_for_prob_predict()
		data_model_dict = {}
		model_dataframe.shuffle_rows()
		x_y_vars = model_dataframe.set_features_and_target1(self.columns_all, self.target_col_name)
		data_model_dict['features'] = x_y_vars[0]
		data_model_dict['target'] = x_y_vars[1]
		# set up training and testing data
		vars_for_train_test = model_dataframe.create_train_and_test_data_x_y_mixer(self.training_percent, data_model_dict['features'],data_model_dict['target'])
		data_model_dict['X_train'] = vars_for_train_test[0]
		data_model_dict['y_train'] = vars_for_train_test[1]
		data_model_dict['X_test'] = vars_for_train_test[2]
		data_model_dict['y_test'] = vars_for_train_test[3]
		return data_model_dict

		# can prob make this kwargs class variables
	def predict_prob_model_full(self):
		# vars
		data = self._set_up_data_for_models() 
		# start prediction instace 
		predictions_instance = RegressionCombined(data['features'], data['target'], self.kfold_dict, data['X_train'], data['X_test'], data['y_train'], data['y_test'])
		predictions_results = predictions_instance.regression_probs_model(param_dict_logistic=self.param_dict_logistic, param_dict_decision_tree=self.param_dict_decision_tree, param_dict_neural_network=self.param_dict_neural_network)
		return predictions_results

	def predict_prob_model_full_fit_parameters(self):
		data = self._set_up_data_for_models() 
		predictions_instance = predictions_instance = RegressionCombined(data['features'], data['target'], self.kfold_dict, data['X_train'], data['X_test'], data['y_train'], data['y_test'])
		predictions_results = predictions_instance.regression_probs_model_full_paramter_fit(param_dict_logistic_array=self.param_dict_logistic_array, param_dict_decision_tree_array=self.param_dict_decision_tree_array, param_dict_neural_network_array=self.param_dict_neural_network_array )
		return predictions_results

	# iterate over self.columns_all to return different combinations of columns_all
	# to run models on
	def _cycle_vars(self):
		cols_array = []
		cols = self.columns_all
		combos_array = []
		dict = {}
		y = 0
		for x in range(0, len(cols)+1):
			for subset in itertools.combinations(cols, x):
				#print(subset)
				combos_array.append(subset)
				dict[y] = subset
				y +=1 	
		return dict

	# this is to take into a different var for feature column list
	def _set_up_data_for_models_test(self, columns_all):
		model_dataframe = self._set_up_data_for_prob_predict()
		data_model_dict = {}
		model_dataframe.shuffle_rows()
		x_y_vars = model_dataframe.set_features_and_target1(columns_all, self.target_col_name)
		data_model_dict['features'] = x_y_vars[0]
		data_model_dict['target'] = x_y_vars[1]
		# set up training and testing data
		vars_for_train_test = model_dataframe.create_train_and_test_data_x_y_mixer(self.training_percent, data_model_dict['features'],data_model_dict['target'])
		data_model_dict['X_train'] = vars_for_train_test[0]
		data_model_dict['y_train'] = vars_for_train_test[1]
		data_model_dict['X_test'] = vars_for_train_test[2]
		data_model_dict['y_test'] = vars_for_train_test[3]
		return data_model_dict

	# this takes in the user input columns all and returns a dict with all the
	# different combinations of those columns. thesevalues can be fed into
	# _set_up_data_for_models_test
	def _cycle_vars_dict(self):
		cols = self.columns_all
		#print(cols)
		cols.remove(self.target_col_name)
		var_combo_dict = {}
		y = 0
		for x in range(0, len(cols)+1):
			for subset in itertools.combinations(cols, x):
				subset = list(subset)
				subset.append(self.target_col_name)
				var_combo_dict[y] = subset
				y +=1
		return var_combo_dict 
		#cols_all1 = var_combo_dict[3]
		#print(cols_all1)
		#print(type(cols_all1))
		#data = self._set_up_data_for_models_test(cols_all1)		 	
		#predictions_instance = RegressionCombined(data['features'], data['target'], self.kfold_dict, data['X_train'], data['X_test'], data['y_train'], data['y_test'])
		#predictions_results = predictions_instance.regression_probs_model(param_dict_logistic=self.param_dict_logistic, param_dict_decision_tree=self.param_dict_decision_tree, param_dict_neural_network=self.param_dict_neural_network)
		#return predictions_results

	# method to iteerate over the different combos of vars and then spit out certain scores
	def cycle_vars_return_desired_output(self):
		#print(self.target_col_name)
		#print(type(self.target_col_name))
		#print(self.columns_all)
		vars_combo = self._cycle_vars_dict()
		dict_score_combos = {}
		for x in vars_combo.values():
			if len(x) > 1:
				print(type(x))
				print(x)
				#y =str(x)
				#print(type(y))
				#print(y)
				data = self._set_up_data_for_models_test(x)
				predictions_instance = RegressionCombined(data['features'], data['target'], self.kfold_dict, data['X_train'], data['X_test'], data['y_train'], data['y_test'])
				predictions_results = predictions_instance.regression_probs_model(param_dict_logistic=self.param_dict_logistic, param_dict_decision_tree=self.param_dict_decision_tree, param_dict_neural_network=self.param_dict_neural_network)
				dict_score_combos[str(x)] = predictions_results
		return dict_score_combos
			#print(dict_score_combos)

	def user_output_model(self):
		data = self._set_up_data_for_models() 
		# start prediction instace 
		predictions_instance = RegressionCombined(data['features'], data['target'], self.kfold_dict, data['X_train'], data['X_test'], data['y_train'], data['y_test'], user_input_for_model_output=self.user_input_for_model_output)
		output = predictions_instance.classification_unifying_model()
		print(output)


# info for bikes
file_location = '/home/mike/Documents/coding_all/machine_predict/hour.csv'
df_bike = pd.read_csv(file_location)
columns_to_drop_bike = ['casual', 'registered', 'dtedat']
columns_all_bike = ['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt_binary']
columns_all_bike_test = ['workingday','temp', 'cnt_binary']
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
nnl_params_bike = {'hidden_layer_sizes':(10, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':random_state_bike}
kfold_dict = {'n_splits':10, 'random_state':random_state_bike, 'shuffle':False}
model_score_dict = {'logistic':{'error_metric':'roc_auc_score', 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
user_optmize_input = ['class', 'constant', 'train', model_score_dict]
# bike model....
#bike_predict = MachinePredictModel(df_bike, columns_all_bike, random_state_bike, training_percent_bike, kfold_number_bike, target_bike, cols_to_drop=columns_to_drop_bike,set_multi_class=set_multi_class_bike, target_change_bin_dict=create_target_dict_bike, kfold_dict=kfold_dict)
#bike_predict._set_up_data_for_prob_predict()
#results = bike_predict.predict_prob_model(param_dict_logistic=logistic_regression_params_bike, param_dict_decision_tree=decision_tree_params_bike,param_dict_neural_network=nnl_params_bike)
# bike model for optimizing 
# range of values in dict form for parameters
decision_tree_array_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':[None, 'balanced'], 'random_state':[random_state_bike]}
logistic_regression_array_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1], 'random_state':[random_state_bike]}
neural_net_array_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[True], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08], 'random_state':[random_state_bike]}
# optimize model 
#bike_predict.predict_prob_model_fit_parameters(training_percent_bike, kfold_number_bike, target_bike, param_dict_logistic_array=logistic_regression_array_vars, param_dict_decision_tree_array=decision_tree_array_vars, param_dict_neural_network_array=neural_net_array_vars)
#bike_predict.predict_prob_model_fit_parameters(training_percent_bike, kfold_number_bike, target_bike, param_dict_decision_tree_array=decision_tree_array_vars)
# bike models for refractored class
bike_predict = MachinePredictModel(df_bike, columns_all_bike_test, random_state_bike, training_percent_bike, kfold_number_bike, target_bike, cols_to_drop=columns_to_drop_bike,set_multi_class=set_multi_class_bike, target_change_bin_dict=create_target_dict_bike, kfold_dict=kfold_dict, param_dict_logistic=logistic_regression_params_bike, param_dict_decision_tree=decision_tree_params_bike, param_dict_neural_network=nnl_params_bike, param_dict_logistic_array=logistic_regression_array_vars, param_dict_decision_tree_array=decision_tree_array_vars, param_dict_neural_network_array=neural_net_array_vars, user_input_for_model_output=user_optmize_input)
bike_predict.user_output_model()
#bike_predict._set_up_data_for_prob_predict()
#combos1 = bike_predict._cycle_vars_dict()
#combos = bike_predict.cycle_vars_return_desired_output()
#print(combos)
#for x, y in combos.items():
#	print(x)
#	print('_____________________')
#	print(y)
#	print('________________________')
#vars_combo = bike_predict._cycle_vars_dict()
#print(vars_combo)
#for x in vars_combo.values():
	#print(x)
#print(len(vars_combo))
#results2 = bike_predict.predict_prob_model_full()
#print(results2)
"""
results2 = bike_predict.predict_prob_model_full()
print(results2)
print(type(results2))
for x, y in results2.items():
	print('________________')
	print(x)
	print(y)
print('__________________')	
print(results2['dict_results_train_set'])
"""
#results3 = bike_predict.predict_prob_model_full_fit_parameters()
#print(results3)
#columns_all_features_bike = 
#results2 = bike_predict.cycle_vars(columns_all_features_bike, training_percent_bike, kfold_number_bike, target_bike)

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
"""