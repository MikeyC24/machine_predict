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
		# if yes this will cycle thru user params and return wanted scores
		# if no it will just use all the vars given
		self.cycle_vars_user_check = kwargs.get('cycle_vars_user_check', None)
		# if user it will user user entered params for models
		# if no it will find best params and return those.
		self.minimum_feature_count_for_var_cycle = kwargs.get('minimum_feature_count_for_var_cycle', None) 

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
			# this is now working however this is the only equation below that
			# returns the class instance itself instead of a dataframe
			model_dataframe.drop_columns_return_self(self.cols_to_drop)
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


	# can prob make this kwargs class variables
	def predict_prob_model_full(self):
		# vars
		data = self._set_up_data_for_models_test(self.columns_all) 
		# start prediction instace 
		predictions_instance = RegressionCombined(data['features'], data['target'], self.kfold_dict, data['X_train'], data['X_test'], data['y_train'], data['y_test'])
		predictions_results = predictions_instance.regression_probs_model(param_dict_logistic=self.param_dict_logistic, param_dict_decision_tree=self.param_dict_decision_tree, param_dict_neural_network=self.param_dict_neural_network)
		return predictions_results

	def predict_prob_model_full_fit_parameters(self):
		data = self._set_up_data_for_models_test(self.columns_all) 
		predictions_instance = predictions_instance = RegressionCombined(data['features'], data['target'], self.kfold_dict, data['X_train'], data['X_test'], data['y_train'], data['y_test'])
		predictions_results = predictions_instance.regression_probs_model_full_paramter_fit(param_dict_logistic_array=self.param_dict_logistic_array, param_dict_decision_tree_array=self.param_dict_decision_tree_array, param_dict_neural_network_array=self.param_dict_neural_network_array )
		return predictions_results


	# # take in df of cleaned model, return features, target, train and test data in dict
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
		min_count = self.minimum_feature_count_for_var_cycle
		#print(cols)
		cols.remove(self.target_col_name)
		var_combo_dict = {}
		y = 0
		for x in range(0, len(cols)+1):
			for subset in itertools.combinations(cols, x):
				subset = list(subset)
				subset.append(self.target_col_name)
				if min_count == None:
					var_combo_dict[y] = subset
				else:
					if len(subset) > min_count:
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
	def cycle_vars_return_desired_output_specific_model(self):
		#print(self.target_col_name)
		#print(type(self.target_col_name))
		#print(self.columns_all)
		vars_combo = self._cycle_vars_dict()
		dict_score_combos = {}
		for x in vars_combo.values():
			if len(x) > 1:
				#print(x)
				data = self._set_up_data_for_models_test(x)
				predictions_instance = RegressionCombined(data['features'], data['target'], self.kfold_dict, data['X_train'], data['X_test'], data['y_train'], data['y_test'],param_dict_logistic=self.param_dict_logistic, param_dict_decision_tree=self.param_dict_decision_tree, param_dict_neural_network=self.param_dict_neural_network, user_input_for_model_output=self.user_input_for_model_output,param_dict_logistic_array=self.param_dict_logistic_array, param_dict_decision_tree_array=self.param_dict_decision_tree_array, param_dict_neural_network_array=self.param_dict_neural_network_array)
				output = predictions_instance.classification_unifying_model()
				dict_score_combos[str(x)] = output
		return dict_score_combos
			#print(dict_score_combos)

	# this method uses all vars inputed and does the test and train data type inputed
	# from user_input_for_model_output to give one dict of answers, right now it returns
	# data no matter what the error scores are, unsure yet if this should check those
	# should be easy to implment as the return_desired_user_output_from_dict method
	# should work find on this and would just need some if statements
	def user_output_model(self):
		data = self._set_up_data_for_models_test(self.columns_all) 
		# start prediction instace 
		predictions_instance = RegressionCombined(data['features'], data['target'], self.kfold_dict, data['X_train'], data['X_test'], data['y_train'], data['y_test'],param_dict_logistic=self.param_dict_logistic, param_dict_decision_tree=self.param_dict_decision_tree, param_dict_neural_network=self.param_dict_neural_network, user_input_for_model_output=self.user_input_for_model_output,param_dict_logistic_array=self.param_dict_logistic_array, param_dict_decision_tree_array=self.param_dict_decision_tree_array, param_dict_neural_network_array=self.param_dict_neural_network_array)
		output = predictions_instance.classification_unifying_model()
		return output

	#only works on one test, need to set forvariables at top, bascailly all possible options
	# cmpare if user input has it and then return 
	# also need to lead with model 
	# this only works with user input params, need a new metho for optimize params
	# this needs to check if user put in params or if they are optmizied
	def return_desired_user_output_from_dict(self):
		data_wanted = self.cycle_vars_return_desired_output_specific_model()
		class_or_amount = self.user_input_for_model_output[0]
		constant_or_optimize = self.user_input_for_model_output[1] 
		train_method = self.user_input_for_model_output[2]
		model_list = self.user_input_for_model_output[3]
		dict_train_types = ['dict_results_simple', 'dict_results_kfold', 'dict_results_train_set']
		model_types = model_list.keys()
		# variables are the variables used for the regression
		dict_all = {}
		if constant_or_optimize == 'constant':
			for model_use_key in model_types:
				dict_returned  = {}
				for variables,y in data_wanted.items():
					for train_test in dict_train_types:
						if len(y[train_test]) > 0:
							# value is the dictionary contained within the train type
							for value in y.values():
								# key model is the type of test, ir logstic, decision tree
								# model scores is the dict containg all the different error scores of that model
								# this dict1 is too keep the scores passed and then pass to bigger dict
								dict1 = {}
								for key_model, model_scores in value.items():
									for model_item in model_types:
										if key_model == model_item:
											# score key is the error metric name such as roc_auc_score
											# score value is the actual number/value
											score_array = []
											dict = {}
											# score key and values are coming from test
											# model_scores and model_list are coming from user
											for score_key, score_values in model_scores.items():
												for item in model_list[key_model]:
													dict['train_type'] = train_test
													dict['regression_ran'] = key_model
													dict['variables'] = variables
													if item == score_key:
														if score_values > model_list[key_model][item]:
															dict1[score_key] = [score_values]
													dict['scores_passed'] = dict1
												dict_returned[str(variables) + str(key_model)] = dict
			return dict_returned
		else:
			print('user chose optimize')
			# key are the varaibles used
			# value is the dict from returned with data, with key being model used
			for key, value in data_wanted.items():
				dict_multi = {}
				for model_used in model_types:
					if value.get(model_used, False):
						score_data = value[model_used]
						print('key', key)
						print('value', value)
						print('model_used', model_used)
						print(score_data)
						print(model_list.values())
						dict1 = {}	
						for score_value_wanted_key_value in model_list.values():
							for score_value_wanted_key in score_value_wanted_key_value.keys():
								if score_data.get(score_value_wanted_key, False):
									print('score_value_wanted_key', score_value_wanted_key)
									score_from_model = score_data[score_value_wanted_key]
									score_from_user = score_value_wanted_key_value[score_value_wanted_key]
									print('score_from_model', score_from_model)
									print('score_from_user', score_from_user)
									if score_from_model >= score_from_model:
										dict1[score_value_wanted_key] = score_from_model
								dict1['best_params'] = score_data['best_params']
								dict1['best_score'] = score_data['best_score'] 
								dict_all[str(key) +str(model_used)] = dict1
								print(dict_all)
			return dict_all



	# this model takes no inputs from user other than initial vars
	# it makes all decisions based inital inputs
	# needs a new method for when optmizing params are chosen
	def user_full_model(self):
		#model_dataframe = self._set_up_data_for_prob_predict()
		if self.cycle_vars_user_check == 'yes':
			print('cycling vars')
			data_wanted = self.return_desired_user_output_from_dict()
		else:
			print('not cycling vars')
			data_wanted = self.user_output_model()
		return data_wanted

