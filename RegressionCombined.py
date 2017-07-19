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

# this class will combine all regressions
# each regression will take in regression parameters
class RegressionCombined:

	def __init__(self, features, target, kfold_dict, **kwargs):
		self.features = features
		self.target = target
		self.kfold_dict = kfold_dict
		self.param_dict_logistic = kwargs.get('param_dict_logistic', None)
		print(kfold_dict)

	def regression_probs_model(self):
		kfold = KFold(self.features.shape[0], n_folds=self.kfold_dict['n_splits'],random_state=self.kfold_dict['random_state'],shuffle=self.kfold_dict['shuffle'])
		predictions_dict = {}
		# first test logistic regression
		if self.param_dict_logistic is None:
			print('used default params for logistic regression')
			self.param_dict_logistic = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':None, 'random_state':None, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
		else:
			print('used user params for logistic regression')
		print(self.param_dict_logistic)
		reg = LogisticRegression(penalty=self.param_dict_logistic['penalty'], dual=self.param_dict_logistic['dual'], tol=self.param_dict_logistic['tol'], C=self.param_dict_logistic['C'], fit_intercept=self.param_dict_logistic['fit_intercept'], intercept_scaling=self.param_dict_logistic['intercept_scaling'], class_weight=self.param_dict_logistic['class_weight'], random_state=self.param_dict_logistic['random_state'], solver=self.param_dict_logistic['solver'], max_iter=self.param_dict_logistic['max_iter'], multi_class=self.param_dict_logistic['multi_class'], verbose=self.param_dict_logistic['verbose'], warm_start=self.param_dict_logistic['warm_start'], n_jobs=self.param_dict_logistic['n_jobs'])
		reg_instance = reg.fit(self.features, self.target)
		predictions_logistic = reg.predict(self.features)
		#return predictions_dict
		instance_array = [reg_instance]
		variance_values = []
		mse_values = []
		ame_values =[]
		r2_score_values = []
		true_positive_rate_values = []
		false_positive_rate_values = []
		dict = {}
		# this maybe should be a separte method 
		for x in instance_array:
			for train_index, test_index in kfold:
				X_train, X_test = self.features.iloc[train_index], self.features.iloc[test_index]
				y_train, y_test = self.target.iloc[train_index], self.target.iloc[test_index]
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
			dict[str(x)+'_avg_mse'] = np.mean(mse_values)
			dict[str(x)+'_avg_ame'] = np.mean(ame_values)
			dict[str(x)+'_r2_score_values'] = np.mean(r2_score_values)
			dict[str(x)+'_ave_var'] = np.mean(variance_values)
			dict['tpr'] = np.mean(true_positive_rate)
			dict['fpr'] = np.mean(false_positive_rate)
		return dict