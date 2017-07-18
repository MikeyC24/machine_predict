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
#import '/home/mike/Documents/coding_all/machine_predict/ArrangeData.py'  


class DecisionTree:

	# questions for this class
	# 1. the meaning of the consistent .05 probs
	# 2. where to load the x/y vars (class v method)
	# 3. where to initiate the clf classes
	# 4. can the long hand way handle large data
	# 5. should normalized data be used on deicions trees - in this
	# case it did not seem to help
	# try these methods on other data to find error

	def __init__(self, holder_var):
		self.holder_var = holder_var
		#self.random_state = random_state

	# this needs to be fixed as random state was taken out of calss var
	def basic_tree(self, X_train, y_train, X_test, y_test):
		clf = DecisionTreeClassifier(random_state=self.random_state)
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		error = roc_auc_score(y_test, predictions)
		return error

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

	def basic_tree_with_vars(self, X_train, y_train, X_test, y_test, **kwargs):
		param_dict = kwargs.get('param_dict_decision_tree', None)
		if param_dict is None:
			print('used default params for decision tree')
			param_dict = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':None, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':None, 'presort':False}
		else:
			print('used user params for decision tree')
		print(param_dict)
		clf = DecisionTreeClassifier(criterion=param_dict['criterion'], splitter=param_dict['splitter'], max_depth=param_dict['max_depth'], min_samples_split=param_dict['min_samples_split'], min_samples_leaf=param_dict['min_samples_leaf'], min_weight_fraction_leaf=param_dict['min_weight_fraction_leaf'], max_features=param_dict['max_features'], random_state=param_dict['random_state'], max_leaf_nodes=param_dict['max_leaf_nodes'], min_impurity_split=param_dict['min_impurity_split'], class_weight=param_dict['class_weight'], presort=param_dict['presort'])
		#clf = DecisionTreeClassifier(criterion=param_dict['criterion'], splitter=param_dict['splitter'] )
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		y = y_test
		tpr_fpr_rates = self._create_false_pos_and_false_neg(predictions, y)
		dict = {}
		dict['roc_auc_score'] = roc_auc_score(y, predictions)
		dict['mse'] = mean_squared_error(y, predictions)
		dict['mae'] = mean_absolute_error(y, predictions)
		dict['r2_score'] = r2_score(y, predictions)
		dict['variance'] = np.var(predictions)
		dict['tpr'] = tpr_fpr_rates[0]
		dict['fpr'] = tpr_fpr_rates[1]
		return dict

	# this needs to be fixed as random state was taken out of calss var
	def random_forest_with_vars(self, X_train, y_train, X_test, y_test, \
		min_samples_leaf, n_estimators):
		clf = RandomForestClassifier(random_state=self.random_state, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		auc_error = roc_auc_score(y_test, predictions)
		mse_error = mean_squared_error(y_test, predictions)
		return auc_error, mse_error

"""
new_file_location_csv1 = '/home/mike/Documents/coding_all/data_sets_machine_predict/btc_play_data.csv'
df = pd.read_csv(new_file_location_csv1)
lend_tree_loan_data = '/home/mike/Documents/coding_all/data_sets_machine_predict/cleaned_laons_2007.csv'
df_loans = pd.read_csv(lend_tree_loan_data)

columns = ['Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
target = 'USD_BTC_EX_High'
fold = 10
random_state = 3


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
"""