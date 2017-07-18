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
	def __init__(self, dataframe, columns_all, target, random_state):
		self.dataframe =  dataframe
		self.columns_all = columns_all
		self.target = target
		self.random_state = random_state

	# what this method does, creates binary column to test on, drops cols not needed
	# sets up normalized cols with new names
	# creates month, week, day highs if set, var, time_interval_check=1
	# where should the add ones col be? prob only on models that need it
	# this class should return the models run, errors scores and predictions for
	# each model
	# Kwards vars = time_interval_check, date_un
	def set_up_data_for_prob_predict(self, target_amount, target_col_name, columns_to_drop, training_percent, kfold_number, **kwargs):
		TA = target_amount
		TCN = target_col_name
		CtD = columns_to_drop
		TP = training_percent
		KFN = kfold_number
		time_interval_check = kwargs.get('time_interval_check', None)
		date_unix = kwargs.get('date_unix', None)
		# initiate the data class
		model_dataframe = ArrangeData(self.dataframe)
		#print(time_interval_check, date_unix)
		# check if date_unix = none
		if date_unix != None:
			model_dataframe.format_unix_date(date_unix)
			#return model_dataframe
		if time_interval_check == 1:
			model_dataframe.resample_date(target, 'month_highs_avg', 'M', 'mean')
			model_dataframe.resample_date(target, 'week_highs_avg', 'W', 'mean')
			model_dataframe.resample_date(target, 'day_highs_avg', 'D', 'mean')
			#return model_dataframe
		model_dataframe.overall_data_display(5)


# class vars
file_location = '/home/mike/Documents/coding_all/machine_predict/btc_play_data.csv'
df = pd.read_csv(file_location)
columns = ['EUR_BTC_EX_High', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
columns_all = ['target_new', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
# these columns may or may not be created but target needs to be in col list
target = 'week_highs_avg_change'
random_state = 1
# method vars
target_amount =.05
target_col_name = 'target_new'
columns_to_drop = []
training_percent =.08
kfold_number = 10
#**kwargs
kwarg_dict = {'time_interval_check':1, 'date_unix':'date_unix'}
time_interval_check = 1
date_unix = 'date_unix'

# error
predict = MachinePredictModel(df, columns_all, target, random_state)
predict.set_up_data_for_prob_predict(target_amount, target_col_name, columns_to_drop, training_percent, kfold_number, time_interval_check=1, date_unix='date_unix')