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

# this class will combine all regressions
# each regression will take in regression parameters
class RegressionCombined:

	def __init__(self, features, target, kfold_dict):
		self.features = features
		self.target = target
		self.kfold_dict = kfold_dict
		#self.param_dict_logistic = kwargs.get('param_dict_logistic', None)
		#print(kfold_dict)

	def _get_error_scores_with_tpr_fpr(self, y_target, predictions, **kwargs):
		#self.param_dict_logistic = kwargs.get('param_dict_logistic', None)
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

	# the simple parameters should prob use train data instead
	def regression_probs_model(self, **kwargs):
		param_dict_logistic = kwargs.get('param_dict_logistic', None)
		param_dict_decision_tree = kwargs.get('param_dict_decision_tree', None)
		kfold = KFold(self.features.shape[0], n_folds=self.kfold_dict['n_splits'],random_state=self.kfold_dict['random_state'],shuffle=self.kfold_dict['shuffle'])
		predictions_dict = {}
		# look into if these if else stamtents can be turned into one line
		if param_dict_logistic is None:
			print('used default params for logistic regression')
			param_dict_logistic = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':None, 'random_state':None, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
		else:
			print('used user params for logistic regression')
		if param_dict_decision_tree is None:
			print('used default params for decision tree')
			param_dict_decision_tree = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':None, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':None, 'presort':False}
		else:
			print('used user params for decision tree')
		#print(param_dict_logistic)
		dict_results_kfold = {}
		dict_results_simple = {}
		reg = LogisticRegression(penalty=param_dict_logistic['penalty'], dual=param_dict_logistic['dual'], tol=param_dict_logistic['tol'], C=param_dict_logistic['C'], fit_intercept=param_dict_logistic['fit_intercept'], intercept_scaling=param_dict_logistic['intercept_scaling'], class_weight=param_dict_logistic['class_weight'], random_state=param_dict_logistic['random_state'], solver=param_dict_logistic['solver'], max_iter=param_dict_logistic['max_iter'], multi_class=param_dict_logistic['multi_class'], verbose=param_dict_logistic['verbose'], warm_start=param_dict_logistic['warm_start'], n_jobs=param_dict_logistic['n_jobs'])
		tree = DecisionTreeClassifier(criterion=param_dict_decision_tree['criterion'], splitter=param_dict_decision_tree['splitter'], max_depth=param_dict_decision_tree['max_depth'], min_samples_split=param_dict_decision_tree['min_samples_split'], min_samples_leaf=param_dict_decision_tree['min_samples_leaf'], min_weight_fraction_leaf=param_dict_decision_tree['min_weight_fraction_leaf'], max_features=param_dict_decision_tree['max_features'], random_state=param_dict_decision_tree['random_state'], max_leaf_nodes=param_dict_decision_tree['max_leaf_nodes'], min_impurity_split=param_dict_decision_tree['min_impurity_split'], class_weight=param_dict_decision_tree['class_weight'], presort=param_dict_decision_tree['presort'])
		instance_array = [reg, tree]
		#reg_instance = reg.fit(self.features, self.target)
		for x in instance_array:
			instance = x.fit(self.features, self.target)
			predictions = x.predict(self.features)
			results = self._get_error_scores_with_tpr_fpr(self.target, predictions)
			dict_results_simple[x] = results
		#return dict_results_simple

		 
		for x in instance_array:
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
			dict_results_kfold[str(x)] = dict
		#return dict_results_kfold
		#tree.fit(self.features, self.target)
		#predictions1 =tree.predict(self.features)
		#results1 = self._get_error_scores_with_tpr_fpr(self.target, predictions1)
		return dict_results_simple, dict_results_kfold


"""
decision tree results are wacky 
"""