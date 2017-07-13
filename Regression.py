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
from sklearn.cross_validation import cross_val_predict, KFold
"""
notes
1. lasso regression - https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/
2. logistic regression function with l1 penatly
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
log = LogisticRegression(penalty='l1', solver='liblinear')
log.fit(X, y)
"""

# perform regressions in this class
# much work to be done in this class
class Regression:

	def __init__(self, features, target, random_state):
		self.features = features
		self.target = target
		self.random_state = random_state

	# performs a simple linear regression 
	# can take in multiple columns
	# returns a dict with mse, mae, r2, variance 
	def simple_lin_regres(self, columns, target):
		df = self.dataframe
		cols = columns
		features = df[cols]
		target_var = df[target]
		x = features
		y = target_var
		reg = LinearRegression()
		reg.fit(x, y)
		predictions = reg.predict(x)
		dict ={}
		dict['mse'] = mean_squared_error(y, predictions)
		dict['mae'] = mean_absolute_error(y, predictions)
		dict['r2_score'] = r2_score(y, predictions)
		dict['variance'] = np.var(predictions)
		return(dict)

	# does a linear regression but with k folds test and training variance
	# kfold testing, dont know enought yet if they will be part of testing
	# methods or will be its own method like the simle train and test
	# and just keep spitting our the test variables
	# returns a dict with mse, mae, r2, variance 
	def kfold_test_simple_lin_regres(self, columns, target, fold, random_state):
		df = self.dataframe
		features = df[columns]
		target = df[target]
		variance_values = []
		mse_values = []
		ame_values =[]
		r2_score_values = []
		dict ={}
		n = len(df)
		# kfold instance
		kf = KFold(n, fold, shuffle=True, random_state = random_state)
		#iterate over the k fold
		for train_index, test_index in kf:
			#trainging and test sets
			# Make predictions on training set.
			x_train, x_test = features.iloc[train_index], features.iloc[test_index]
			y_train, y_test = target.iloc[train_index], target.iloc[test_index]
			lr = LinearRegression()
			lr.fit(x_train, y_train)
			predictions = lr.predict(x_test)
			# Compute MSE and Variance.
			mse = mean_squared_error(y_test, predictions)
			variance = np.var(predictions)
			mae = mean_absolute_error(y_test, predictions)
			r2_scores = r2_score(y_test, predictions)
			#append to array
			variance_values.append(variance)
			mse_values.append(mse)
			ame_values.append(mae)
			r2_score_values.append(r2_scores)
		dict['avg_mse'] = np.mean(mse_values)
		dict['avg_ame'] = np.mean(ame_values)
		dict['r2_score_values'] = np.mean(r2_score_values)
		dict['ave_var'] = np.mean(variance_values)
		return(dict)

	# returns dict of the simple_lin_regress metrics but
	# shows values for each combo of variables
	# this concept will be important later on for optimizing vars
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


	def _create_false_pos_and_false_neg(self, predictions, y_target):
		#df_filter = self.dataframe
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
		return true_positive_rate, false_positive_rate

	# run a logistic regression with kfold cross val predict
	# class_weight is how to weight the logisitc reg
	def logistic_regres_with_kfold_cross_val(self):
		#df = self.dataframe
		#cols = columns
		#features = df[cols]
		#target_var = df[target]
		print(type(self.features))
		print(type(self.target))
		reg = LogisticRegression(class_weight='balanced')
		kf =KFold(self.features.shape[0], random_state=self.random_state)
		reg.fit(self.features, self.target)
		predictions = cross_val_predict(reg, self.features, self.target, cv=kf)
		tpr_fpr_rates = self._create_false_pos_and_false_neg(predictions, self.target)
		dict ={}
		y = self.target
		dict['mse'] = mean_squared_error(y, predictions)
		dict['mae'] = mean_absolute_error(y, predictions)
		dict['r2_score'] = r2_score(y, predictions)
		dict['variance'] = np.var(predictions)
		dict['tpr'] = tpr_fpr_rates[0]
		dict['fpr'] = tpr_fpr_rates[1]
		return(dict)

	def test(self):
		df = self.dataframe
		print(type(df))