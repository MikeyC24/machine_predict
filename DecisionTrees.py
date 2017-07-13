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

class DecisionTree:

	# questions for this class
	# 1. the meaning of the consistent .05 probs
	# 2. where to load the x/y vars (class v method)
	# 3. where to initiate the clf classes
	# 4. can the long hand way handle large data
	# 5. should normalized data be used on deicions trees - in this
	# case it did not seem to help
	# try these methods on other data to find error

	def __init__(self, random_state):
		self.random_state = random_state

	def basic_tree(self, X_train, y_train, X_test, y_test):
		clf = DecisionTreeClassifier(random_state=self.random_state)
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		error = roc_auc_score(y_test, predictions)
		return error

	def basic_tree_with_vars(self, X_train, y_train, X_test, y_test, \
		min_samples_split, max_depth=10):
		clf = DecisionTreeClassifier(random_state=self.random_state, min_samples_split=min_samples_split, max_depth=max_depth)
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		error = roc_auc_score(y_test, predictions)
		return error

	def random_forest_with_vars(self, X_train, y_train, X_test, y_test, \
		min_samples_leaf, n_estimators):
		clf = RandomForestClassifier(random_state=self.random_state, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		auc_error = roc_auc_score(y_test, predictions)
		mse_error = mean_squared_error(y_test, predictions)

		return auc_error, mse_error


# start decision tree here
# X normalized
X = a.dataframe[['ones', 'Transactions_Volume_normalized', 'Number_of_Transactions_normalized', 'LTC_BTC_EX_High_normalized', 'EUR_BTC_EX_High_normalized']].values
# X not normalized
#X = a.dataframe[['ones', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']].values
y = a.dataframe.target_new.values
test_vars = test_project.create_train_and_test_data_x_y_mixer(.07, X, y)
X_train1 = test_vars[0]
y_train1 = test_vars[1]
X_test1 = test_vars[2]
y_test1 = test_vars[3]
detree = DecisionTree(random_state)
basic_tree = detree.basic_tree(X_train1, y_train1, X_test1, y_test1)
print(basic_tree)
min_var_tree = detree.basic_tree_with_vars(X_train1, y_train1, X_test1, y_test1, min_samples_split=13)
print(min_var_tree)
random_forest = detree.random_forest_with_vars(X_train1, y_train1, X_test1, y_test1, n_estimators=150, min_samples_leaf=2)
print(random_forest)



# end decision tree here 
