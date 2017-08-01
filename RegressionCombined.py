import pandas as pd
import numpy as np
import datetime
import time
from itertools import cycle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc,log_loss, precision_score
import matplotlib.pyplot as plt
import operator
# read this of above http://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
from sklearn.model_selection import train_test_split
from scipy import interp
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cross_validation import cross_val_predict, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid, GridSearchCV

# notes need to add a roc area curve for kfold

# this class will combine all regressions
# each regression will take in regression parameters
class RegressionCombined:

	def __init__(self, features, target, kfold_dict, X_train, X_test, y_train, y_test, **kwargs):
		self.features = features
		self.target = target
		self.kfold_dict = kfold_dict
		self.X_train = X_train 
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test
		# this is to determine what test on run and on what traing data 
		self.user_input_for_model_output = kwargs.get('user_input_for_model_output', None)
		self.param_dict_logistic = kwargs.get('param_dict_logistic', None)
		self.param_dict_decision_tree = kwargs.get('param_dict_decision_tree', None)
		self.param_dict_neural_network = kwargs.get('param_dict_neural_network', None)
		self.param_dict_logistic_array = kwargs.get('param_dict_logistic_array', None)
		self.param_dict_decision_tree_array = kwargs.get('param_dict_decision_tree_array', None)
		self.param_dict_neural_network_array = kwargs.get('param_dict_neural_network_array', None)
		

	def _get_error_scores_with_tpr_fpr(self, y_target, predictions):
		#self.param_dict_logistic = kwargs.get('param_dict_logistic', None)
		tp_filter = (predictions == 1) & (y_target == 1)
		tn_filter = (predictions == 0) & (y_target == 0)
		fp_filter = (predictions == 1) & (y_target == 0)
		fn_filter = (predictions == 0) & (y_target == 1)
		tp = len(predictions[tp_filter])
		tn = len(predictions[tn_filter])
		fp = len(predictions[fp_filter])
		fn = len(predictions[fn_filter])
		dict ={}
		if (tp+fn) != 0:
			# this is sensitivity 
			true_positive_rate = tp / (tp+fn)
			dict['sensitivity'] = true_positive_rate
			# false negative rate  
			false_negative_rate = fn/(tp+fn)
			dict['false_negative_rate'] = false_negative_rate
		if (fp + tn) != 0:
			# fall out rate
			false_positive_rate = fp / (fp + tn)
			dict['fallout_rate'] = false_negative_rate
			# this is specificity 
			true_negative_rate = tn/(tn+fp)
			dict['specificty'] = true_negative_rate
		if (fp + tp) != 0:
			# precision or positive predictive value
			precision = tp/(tp+fp)
			dict['precision'] = precision
			#false discovery rate
			false_discovery_rate = fp/(tp+fp)
			dict['false_discovery_rate'] = false_discovery_rate
		if (tn + fn) != 0:
			# negative predictive value
			npv = tn / (tn+fn)
			dict['negative_predictive_value'] = npv
		# overall accuracy
		ACC = (tp+tn) / (tp+fp+fn+tn)
		y = y_target
		dict['roc_auc_score'] = roc_auc_score(y, predictions)
		dict['mse'] = mean_squared_error(y, predictions)
		dict['mae'] = mean_absolute_error(y, predictions)
		dict['r2_score'] = r2_score(y, predictions)
		dict['variance'] = np.var(predictions)
		dict['ACC'] = ACC
		return(dict)

	

	# user input is an array in certain order, some parts are a dict
	def classification_unifying_model(self):
		for x in range(len(self.user_input_for_model_output)):
			print(self.user_input_for_model_output[x])
		if self.user_input_for_model_output is not None:
			# class for class amount for amount
			class_or_amount = self.user_input_for_model_output[0]  
			# constant or optmize
			constant_or_optimize = self.user_input_for_model_output[1]
			# simple, train, kfold
			train_method = self.user_input_for_model_output[2] 
			#this will be a dict that models to run, which error metrics, signifacnt levels
			# range of scores, more to add to this 
			model_list = self.user_input_for_model_output[3]
			
			if class_or_amount == 'class':
				if constant_or_optimize == 'constant':
					data = self.regression_probs_model_with_user_input()
					return data
				elif constant_or_optimize == 'optimize':
					data = self.regression_probs_model_full_paramter_fit_with_user_input()
					return data
				else:
					print('not an accetable constant_or_optimize var')
			elif class_or_amount == 'amount':
				print('still working on this part')
			else:
				print('not an acceptable class_or_amount var')
		else:
			print('no user inputs')



	# the simple parameters should prob use train data instead
	def regression_probs_model_with_user_input(self):
		train_method = self.user_input_for_model_output[2]
		print(train_method)
		model_list = self.user_input_for_model_output[3]
		kfold = KFold(self.features.shape[0], n_folds=self.kfold_dict['n_splits'],random_state=self.kfold_dict['random_state'],shuffle=self.kfold_dict['shuffle'])
		# look into if these if else stamtents can be turned into one line
		if self.param_dict_logistic is None:
			print('used default params for logistic regression')
			param_dict_logistic = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':None, 'random_state':None, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
		else:
			print('used user params for logistic regression')
			param_dict_logistic= self.param_dict_logistic
		if self.param_dict_decision_tree is None:
			print('used default params for decision tree')
			param_dict_decision_tree = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':None, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':None, 'presort':False}
		else:
			print('used user params for decision tree')
			param_dict_decision_tree = self.param_dict_decision_tree
		if self.param_dict_neural_network is None:
			print('used default params for neural network')
			param_dict_neural_network = {'hidden_layer_sizes':(100, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':None}
		else:
			print('used user params for neural network')
			param_dict_neural_network = self.param_dict_neural_network
		#print(param_dict_logistic)
		dict_results_kfold = {}
		dict_results_simple = {}
		dict_results_train_set = {}
		dict_all = {}
		instance_array_var = list(model_list.keys())
		instance_array_classes = []
		instance_array_names = []
		for item in instance_array_var:
			if item == 'LogisticRegress':
				logistic = LogisticRegression(penalty=param_dict_logistic['penalty'], dual=param_dict_logistic['dual'], tol=param_dict_logistic['tol'], C=param_dict_logistic['C'], fit_intercept=param_dict_logistic['fit_intercept'], intercept_scaling=param_dict_logistic['intercept_scaling'], class_weight=param_dict_logistic['class_weight'], random_state=param_dict_logistic['random_state'], solver=param_dict_logistic['solver'], max_iter=param_dict_logistic['max_iter'], multi_class=param_dict_logistic['multi_class'], verbose=param_dict_logistic['verbose'], warm_start=param_dict_logistic['warm_start'], n_jobs=param_dict_logistic['n_jobs'])
				instance_array_classes.append(logistic)
				instance_array_names.append('logistic')
			elif item == 'DecisionTreeCla':
				decision_tree = DecisionTreeClassifier(criterion=param_dict_decision_tree['criterion'], splitter=param_dict_decision_tree['splitter'], max_depth=param_dict_decision_tree['max_depth'], min_samples_split=param_dict_decision_tree['min_samples_split'], min_samples_leaf=param_dict_decision_tree['min_samples_leaf'], min_weight_fraction_leaf=param_dict_decision_tree['min_weight_fraction_leaf'], max_features=param_dict_decision_tree['max_features'], random_state=param_dict_decision_tree['random_state'], max_leaf_nodes=param_dict_decision_tree['max_leaf_nodes'], min_impurity_split=param_dict_decision_tree['min_impurity_split'], class_weight=param_dict_decision_tree['class_weight'], presort=param_dict_decision_tree['presort'])
				instance_array_classes.append(decision_tree)
				instance_array_names.append('decision_tree')
			elif item == 'MLPClassifier(a':
				neural_network = nnl = MLPClassifier(hidden_layer_sizes=param_dict_neural_network['hidden_layer_sizes'], activation=param_dict_neural_network['activation'], solver=param_dict_neural_network['solver'], alpha=param_dict_neural_network['alpha'], batch_size=param_dict_neural_network['batch_size'], learning_rate=param_dict_neural_network['learning_rate'], learning_rate_init=param_dict_neural_network['learning_rate_init'], power_t=param_dict_neural_network['power_t'], max_iter=param_dict_neural_network['max_iter'], shuffle=param_dict_neural_network['shuffle'], random_state=param_dict_neural_network['random_state'], tol=param_dict_neural_network['tol'], verbose=param_dict_neural_network['verbose'], warm_start=param_dict_neural_network['warm_start'], momentum=param_dict_neural_network['momentum'], nesterovs_momentum=param_dict_neural_network['nesterovs_momentum'], early_stopping=param_dict_neural_network['early_stopping'], validation_fraction=param_dict_neural_network['validation_fraction'], beta_1=param_dict_neural_network['beta_1'], beta_2=param_dict_neural_network['beta_2'], epsilon=param_dict_neural_network['epsilon'])
				instance_array_classes.append(neural_network)
				instance_array_names.append('neural_network')
			else:
				print('none of those regressions, classifications or models are supported')

		reg = LogisticRegression(penalty=param_dict_logistic['penalty'], dual=param_dict_logistic['dual'], tol=param_dict_logistic['tol'], C=param_dict_logistic['C'], fit_intercept=param_dict_logistic['fit_intercept'], intercept_scaling=param_dict_logistic['intercept_scaling'], class_weight=param_dict_logistic['class_weight'], random_state=param_dict_logistic['random_state'], solver=param_dict_logistic['solver'], max_iter=param_dict_logistic['max_iter'], multi_class=param_dict_logistic['multi_class'], verbose=param_dict_logistic['verbose'], warm_start=param_dict_logistic['warm_start'], n_jobs=param_dict_logistic['n_jobs'])
		tree = DecisionTreeClassifier(criterion=param_dict_decision_tree['criterion'], splitter=param_dict_decision_tree['splitter'], max_depth=param_dict_decision_tree['max_depth'], min_samples_split=param_dict_decision_tree['min_samples_split'], min_samples_leaf=param_dict_decision_tree['min_samples_leaf'], min_weight_fraction_leaf=param_dict_decision_tree['min_weight_fraction_leaf'], max_features=param_dict_decision_tree['max_features'], random_state=param_dict_decision_tree['random_state'], max_leaf_nodes=param_dict_decision_tree['max_leaf_nodes'], min_impurity_split=param_dict_decision_tree['min_impurity_split'], class_weight=param_dict_decision_tree['class_weight'], presort=param_dict_decision_tree['presort'])
		nnl = MLPClassifier(hidden_layer_sizes=param_dict_neural_network['hidden_layer_sizes'], activation=param_dict_neural_network['activation'], solver=param_dict_neural_network['solver'], alpha=param_dict_neural_network['alpha'], batch_size=param_dict_neural_network['batch_size'], learning_rate=param_dict_neural_network['learning_rate'], learning_rate_init=param_dict_neural_network['learning_rate_init'], power_t=param_dict_neural_network['power_t'], max_iter=param_dict_neural_network['max_iter'], shuffle=param_dict_neural_network['shuffle'], random_state=param_dict_neural_network['random_state'], tol=param_dict_neural_network['tol'], verbose=param_dict_neural_network['verbose'], warm_start=param_dict_neural_network['warm_start'], momentum=param_dict_neural_network['momentum'], nesterovs_momentum=param_dict_neural_network['nesterovs_momentum'], early_stopping=param_dict_neural_network['early_stopping'], validation_fraction=param_dict_neural_network['validation_fraction'], beta_1=param_dict_neural_network['beta_1'], beta_2=param_dict_neural_network['beta_2'], epsilon=param_dict_neural_network['epsilon'])
		#instance_array = [reg, tree, nnl]
		#instance_array_name = ['reg_model', 'tree_model', 'nnl_model']
		#reg_instance = reg.fit(self.features, self.target)
		if train_method == 'simple':
			for x in instance_array_classes:
				#for y in range(len(instance_array_classes)):
				instance = x.fit(self.features, self.target)
				predictions = x.predict(self.features)
				results = self._get_error_scores_with_tpr_fpr(self.target, predictions)
				dict_results_simple[str(x)[:15]] = results
				#return dict_results_simple
		elif train_method == 'train':
			for x in instance_array_classes:
				#for y in range(len(instance_array_classes)):
				instance = x.fit(self.X_train, self.y_train)
				predictions = x.predict(self.X_test)
				results = self._get_error_scores_with_tpr_fpr(self.y_test, predictions)
				dict_results_train_set[str(x)[:15]] = results
				#dict_results_train_set[str(x)[:15]] = results
		elif train_method == 'kfold':
			for x in instance_array_classes:
				#for y in range(len(instance_array_classes)):
				dict ={}
				variance_values = []
				mse_values = []
				ame_values =[]
				r2_score_values = []
				true_positive_rate_values = []
				false_positive_rate_values = []
				for train_index, test_index in kfold:
					X_train, X_test = self.features.iloc[train_index], self.features.iloc[test_index]
					y_train, y_test = self.target.iloc[train_index], self.target.iloc[test_index]
					instance = x.fit(self.features, self.target)
					predictions = x.predict(X_test)
					mse = mean_squared_error(y_test, predictions)
					variance = np.var(predictions)
					mae = mean_absolute_error(y_test, predictions)
					r2_scores = r2_score(y_test, predictions)
					#append to array 
					variance_values.append(variance)
					mse_values.append(mse)
					ame_values.append(mae)
					r2_score_values.append(r2_scores)
					tp_filter = (predictions == 1) & (y_test == 1)
					tn_filter = (predictions == 0) & (y_test == 0)
					fp_filter = (predictions == 1) & (y_test == 0)
					fn_filter = (predictions == 0) & (y_test == 1)
					tp = len(predictions[tp_filter])
					tn = len(predictions[tn_filter])
					fp = len(predictions[fp_filter])
					fn = len(predictions[fn_filter])
					true_positive_rate = tp / (tp+fn)
					false_positive_rate = fp / (fp + tn)
					true_positive_rate_values.append(true_positive_rate)
					false_positive_rate_values.append(false_positive_rate)
				dict['avg_mse'] = np.mean(mse_values)
				dict['avg_ame'] = np.mean(ame_values)
				dict['r2_score_values'] = np.mean(r2_score_values)
				dict['ave_var'] = np.mean(variance_values)
				dict['tpr'] = np.mean(true_positive_rate)
				dict['fpr'] = np.mean(false_positive_rate)
				#dict_results_kfold[instance_array_names[y]] = dict
				dict_results_kfold[str(x)[:15]] = dict
		else:
			print('none of those data training methods are supported')
		dict_all['dict_results_simple'] = dict_results_simple
		dict_all['dict_results_kfold'] = dict_results_kfold
		dict_all['dict_results_train_set'] = dict_results_train_set
		return dict_all

	def regression_probs_model_full_paramter_fit_with_user_input(self):
		#param_dict_logistic_array = kwargs.get('param_dict_logistic_array', None)
		#param_dict_decision_tree_array = kwargs.get('param_dict_decision_tree_array', None)
		#param_dict_neural_network_array = kwargs.get('param_dict_neural_network_array', None)
		model_list = self.user_input_for_model_output[3]
		dtree = DecisionTreeClassifier()
		reg = LogisticRegression()
		nnl = MLPClassifier()
		dict_results_parameter_fit = {}
		#print('below are inputs for param grid search, in order oglog, tree, nnl')
		#print(self.param_dict_logistic_array)
		#print(self.param_dict_decision_tree_array)
		#print(self.param_dict_neural_network_array)
		if 'LogisticRegress' in model_list:
			print('doing log regress')
			clf = GridSearchCV(reg, self.param_dict_logistic_array)
			clf.fit(self.X_train, self.y_train)
			predictions = clf.predict(self.X_test)
			error_score = self._get_error_scores_with_tpr_fpr(self.y_test, predictions)
			error_score['best_score'] = clf.best_score_
			error_score['best_params'] = clf.best_params_
			dict_results_parameter_fit['LogisticRegress'] = error_score
		if 'DecisionTreeCla' in model_list:
			print('doing decision tree')
			clf = GridSearchCV(dtree, self.param_dict_decision_tree_array)
			clf.fit(self.X_train, self.y_train)
			predictions = clf.predict(self.X_test)
			error_score = self._get_error_scores_with_tpr_fpr(self.y_test, predictions)
			error_score['best_score'] = clf.best_score_
			error_score['best_params'] = clf.best_params_
			dict_results_parameter_fit['DecisionTreeCla'] = error_score
		if 'MLPClassifier(a' in model_list:
			print('doing nnl')
			clf = GridSearchCV(nnl, self.param_dict_neural_network_array)
			clf.fit(self.X_train, self.y_train)
			predictions = clf.predict(self.X_test)
			error_score = self._get_error_scores_with_tpr_fpr(self.y_test, predictions)
			error_score['best_score'] = clf.best_score_
			error_score['best_params'] = clf.best_params_
			dict_results_parameter_fit['MLPClassifier'] = error_score
		return dict_results_parameter_fit

