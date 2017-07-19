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

#new_file_location_csv = '/home/mike/Documents/coding_all/cyper_cur/btc_play_data.csv'
#new_file_location_csv1 = '/home/mike/Documents/coding_all/cyper_cur/btc_play_data.csv'
#df = pd.read_csv(new_file_location_csv1)


class ArrangeData:

	# goal of class
	# display some basic data (no graphs)
	# arrange/format/edit dataframe to perform on
	# return type should always be df
	# it takes in a df and spits one out

	# the regression in this class needs to be refactored out 
	# idea to deal with large data, maybe have some methods that
	# can read into a sql db, select parameters and return
	# a temp csv to parse, aaron has an example of this
	# need to add a method that can pull wanted columns from multiple dbs
	# and return a dataframe 
	# also want a column that looks at tpr and tfr etc... to optimize on

	def __init__(self, dataframe):
		self.dataframe = dataframe

	# display column names, shape, and x number of rows you want
	# needs a better display message 
	def overall_data_display(self, headnum):
		df = self.dataframe
		column_list = list(df.columns.values)
		df_total = df.shape
		column_total = df.columns.shape[0]
		print(column_list, df_total, column_total)
		# make this hve a better use friendly output message
		print(df.head(headnum))

	# print all values in a coulmn
	def print_column_all(self, column_name):
		df = self.dataframe
		x = df.loc[:,column_name]
		print(x)

	# print all columns down to number of rows specified
	def print_headers(self, count):
		df = df = self.dataframe
		print(df.head(count))

	# convert a column to a new datetime column
	# dates need to be in datetime for pandas functionality
	def convert_to_datetime(self, column_name_to_convert, new_column_name):
		df = self.dataframe
		df[new_column_name] = pd.to_datetime(df[column_name])
		return df

	# this is not working
	# goal is to be able to shift a column as some variables we want to
	# create will need this (i.e. rolling averages)
	def shift_data(self, column_name, periods):
		x = df.loc[:,column_name]
		x.shift(periods=periods)
		return df

	# drop row from df
	def drop_rows(self, number):
		df = self.dataframe
		df = df.drop(df.index[:number], inplace=True)
		return df

	# shuffle rows
	# this is needed to set up data sometimes for testing
	def shuffle_rows(self):
		df = self.dataframe
		shuffled_rows = np.random.permutation(df.index)
		df = df.loc[shuffled_rows,:]
		return df

	# returns a new column with the rolling average for the column 
	# given and the period/frequency of the window
	def rolling_average(self, column_name, periods):
		df = self.dataframe
		x = df[column_name]
		y = df['shifted'] = x.shift(-1)
		df['MA' + str(periods)] = y.rolling(window =periods).mean()
		return df

	# returns a new column with the rolling std for the column
	# given and period/ frequency given
	def rolling_std(self, column_name, periods):
		df = self.dataframe
		x = df[column_name]
		y = df['shifted'] = x.shift(-1)
		df['STD' + str(periods)] = y.rolling(window =periods).std()
		return df

	# http://blog.mathandpencil.com/group-by-datetimes-in-pandas
	# above link is for grouping by weeks, months etc from unix and datatimes
	# https://stackoverflow.com/questions/26646191/pandas-groupby-month-and-year 
	# above link also has rly good info on this 
	# this is not working
	def group_by_week(self, column_name):
		df = self.dataframe
		#df[column_name] = pd.to_datetime(df[column_name])
		"""
		#print(type(df[column_name]))
		df.groupby(pd.TimeGrouper('M'))
		return df
		x = pd.to_datetime(df[column_name])
		per = x.dt.to_period('M')
		g = df.groupby(per)
		df['month_groups'] = g.sum()
		return df
		"""
		#df['date_minus_time'] = df["_id"].apply( lambda df : 
		#datetime.datetime(year=df.year, month=df.month, day=df.day))	
		#df.set_index(df["date_minus_time"],inplace=True)
		df['new_date'] = df[column_name].apply( lambda df: datetime.datetime(year=df.year, month=df.month, day=df.day))
		df.set_index(df['new_date'], inplace=True)
		return df

	# takes in a column and creates a new column from that in binary for
	# 1 or 0, yes or no, for given target.
	# example all values above value=5 return 1 else 0
	def set_binary(self, column_name_old, column_name_new, value):
		df = self.dataframe
		y = float(value)
		df[column_name_new] = np.where(df[column_name_old] >= y, 1, 0)
		return df

	# keys here are column_name_old, column_name_name, value
	def set_binary_from_dict(self, dict_vars):
		df = self.dataframe
		#y = float(value)
		#df[column_name_new] = np.where(df[column_name_old] >= y, 1, 0)
		print(len(dict_vars))
		print(len(dict_vars.values()))
		print(len(dict_vars.keys()))
		count1 = int(len(dict_vars.values()) / len(dict_vars.keys()))
		print(count1)
		count = sum(len(v) for v in dict_vars.values())
		range_count = int(count / len(list(dict_vars.keys())))
		print(count)
		print(range_count)
		for x in range(range_count):
			print(x)
			print(dict_vars['value'][x])
			print(dict_vars['column_name_new'][x])
			print(dict_vars['column_name_old'][x])
			y = float(dict_vars['value'][x])
			df[dict_vars['column_name_new'][x]] = np.where(df[dict_vars['column_name_old'][x]] >= y, 1, 0)
		return df


	# takes in a column name, 4 target values and returns a 5 class
	# scale 1-5 
	# some thoughts on this....
	# 1. can do this be dine for any number of classes?
	# 2. have it output 1-5, strings or both
	def set_multi_class(self, column_use, value_low, value_low_mid, value_high_mid, value_high, column_name_new):
		df = self.dataframe
		x = column_use
		df[column_name_new] = 0
		mask = df[x] < value_low
		mask2 = (df[x] < value_low_mid) & (df[x] >= value_low )
		mask3 = (df[x] < value_high_mid) & (df[x] >= value_low_mid)
		mask4 = (df[x] >= value_high_mid) & (df[x] < value_high)
		mask5 = df[x] > value_high
		df.loc[mask, column_name_new] = 1
		df.loc[mask2, column_name_new] = 2
		df.loc[mask3, column_name_new] = 3
		df.loc[mask4, column_name_new] = 4
		df.loc[mask5, column_name_new] = 5
		#df.loc[mask, 'target_5_class'] = 'less than '+ str(value_low)
		#df.loc[mask2, 'target_5_class'] = 'between ' +str(value_low) + ' ' + str(value_low_mid)
		#df.loc[mask3, 'target_5_class'] = 'between ' +str(value_low_mid) + ' ' + str(value_high_mid)
		#df.loc[mask4, 'target_5_class'] = 'between ' +str(value_high_mid) + ' ' + str(value_high)
		#df.loc[mask5, 'target_5_class'] = 'greater than ' + str(value_high)
		return df

	# takes an in array in the below specified order
	# column_use, value_low, value_low_mid, value_high_mid, value_high, column_name_new
	def set_multi_class_array(self, array_vars):
		column_use = array_vars[0]  
		value_low = array_vars[1]  
		value_low_mid = array_vars[2]  
		value_high_mid = array_vars[3]  
		value_high = array_vars[4]
		column_name_new = array_vars[5] 
		df = self.dataframe
		x = column_use
		df[column_name_new] = 0
		mask = df[x] < value_low
		mask2 = (df[x] < value_low_mid) & (df[x] >= value_low )
		mask3 = (df[x] < value_high_mid) & (df[x] >= value_low_mid)
		mask4 = (df[x] >= value_high_mid) & (df[x] < value_high)
		mask5 = df[x] > value_high
		df.loc[mask, column_name_new] = 1
		df.loc[mask2, column_name_new] = 2
		df.loc[mask3, column_name_new] = 3
		df.loc[mask4, column_name_new] = 4
		df.loc[mask5, column_name_new] = 5
		#df.loc[mask, 'target_5_class'] = 'less than '+ str(value_low)
		#df.loc[mask2, 'target_5_class'] = 'between ' +str(value_low) + ' ' + str(value_low_mid)
		#df.loc[mask3, 'target_5_class'] = 'between ' +str(value_low_mid) + ' ' + str(value_high_mid)
		#df.loc[mask4, 'target_5_class'] = 'between ' +str(value_high_mid) + ' ' + str(value_high)
		#df.loc[mask5, 'target_5_class'] = 'greater than ' + str(value_high)
		return df

	# creates a columns of all ones
	# this is needed for certain models like neural networks
	def set_ones(self):
		df = self.dataframe
		df['ones'] = np.ones(df.shape[0])
		return df

	# creates dummy variables
	# have option to append new column, also can drop old one and both
	# can be used for something like multi class classification
	def dummy_variables(self, column_name, prefix, append=1, drop=0):
		df = self.dataframe
		dummy_var = pd.get_dummies(df[column_name], prefix=prefix)
		if append == 1 and drop != 1:
			df = pd.concat([df, dummy_var], axis=1)
			return df
		elif append == 1 and drop == 1:
			df = pd.concat([df, dummy_var], axis=1)
			df = df.drop(column_name, axis=1)
			return df
		else: 
			return dummy_var
	
	# normalize numerical data columns
	# good for mahcine learning and/or
	# different data sets have very large ranges
	# this can take in an array of columns
	def normalize_new_column(self, columns_array):
		df = self.dataframe
		#result = df.copy()
		for feature_name in columns_array:
			max_value = df[feature_name].max()
			min_value = df[feature_name].min()
			df[str(feature_name)+'_normalized'] = (df[feature_name] - min_value) / (max_value - min_value)
		return df

	# returns date, need to put in date column, to date time and 
	# then indexes the column
	# look up why it is indexed, it may just be to support the 
	# resample method
	def format_unix_date(self, column_name):
		df = self.dataframe
		x= df[column_name]
		df['Datetime'] = pd.to_datetime(x, unit='s')
		df.index=df['Datetime']
		return df

	# return a dataframe with new columns based on time intervals
	# old column is the sample you want to use
	# new column is the new name, frew is one letter, m,d,w,
	# how can be mean or sum or similar 
	# this method groups the data points by the time interval 
	# so W mean each number is grouped by week
	def resample_date(self, column_old, column_new, freq, method):
		df = self.dataframe
		df[column_new] = df[column_old].resample(freq, how=method)
		return df

	#top method will be depreciated, this needs to be updated for .20 version
	def resample_date_new_way(self, column_old, column_new, freq, method):
		df = self.dataframe
		df[column_new] = df[column_old].resample(freq, how=method)
		return df


	# the number has looks a certain number of days back, need to group like above
	# takes in a column and returns a new one of returns based on time period
	# this may interplay with some of the later stuff in the above method
	# default freq here is 1
	def time_period_returns(self, column_name_old, column_name_new, freq=1):
		df = self.dataframe
		prices = df[column_name_old]
		#print(type(prices))
		df[column_name_new] = prices.pct_change(freq)
		return df

	# take in a dict of column_name_old, column_name_new, fred
	# this should be editied to throw backan error if the number of vars
	# is in the wrong ratio
	# also keys need to have right name
	def time_period_returns_dict(self, dict_vars):
		df = self.dataframe
		count = sum(len(v) for v in dict_vars.values())
		range_count = int(count / len(list(dict_vars.keys())))
		for x in range(range_count):
			prices = df[dict_vars['column_name_old'][x]]
			#print(type(prices))
			df[dict_vars['column_name_new'][x]] = prices.pct_change(dict_vars['freq'][x])
		return df


	def drop_col_by_percent_info_has(self, percent):
		df = self.dataframe
		len_col = df.shape[0]
		array_missing = []
		for x in list(df):
			missing_col_values = df[x].isnull().sum()
			if missing_col_values > (len_col * percent):
				array_missing.append(x)
		df = df.drop(array_missing, axis = 1)
		df = df.dropna()
		return df

	def drop_columns(self, column_array):
		df = self.dataframe
		for x in df:
			df = df.drop([x], axis = 1)
		return df

	def convert_to_num(self, columns):
		pass
		# get ride of all non num with regex .str.rstrip('%').
		#convert to num astype('float')

	# makes a new column with 1s when ma1 is higher than ma2
	# also can be looked at when ma1 passes ma2 
	#df[str(x)+'_normalized'] = df[x].apply(func, axis=1)
	def make_rolling_avg_pass_point_binary(self, ma1, ma2):
		df = self.dataframe
		df[str(ma1) + 'passes' + str(ma2)] = np.where(df[ma1] > df[ma2], 1, 0)
		#if df[ma1] > df[m2]:
		#	df[str(ma1) + 'passes' + str(ma2)] = 1
		#else:
		#	df[str(ma1) + 'passes' + str(ma2)] = 0
		#return df

	# creates a train and test set from a given X and Y
	# some thoughts on this
	# 1. will prob end in testing class
	# 2. dont know yet if this should take in x,y or take in a df
	# right now this returns 4 variables to use for testing
	def create_train_and_test_data_x_y_mixer(self, percent, X, y):
		df = self.dataframe
		highest_train_row = int(df.shape[0]*(percent))
		X_train = X[:highest_train_row]
		y_train = y[:highest_train_row]
		X_test = X[highest_train_row:]
		y_test = y[highest_train_row:]
		return X_train, y_train, X_test, y_test

	# y_target should be the series of the column, not just name
	# 
	def create_false_pos_and_false_neg(self,prediction_column, y_target, column_name_end):
		df_filter = self.dataframe
		tp_filter = (prediction_column == 1) & (y_target == 1)
		tn_filter = (prediction_column == 0) & (y_target == 0)
		fp_filter = (prediction_column == 1) & (y_target == 0)
		fn_filter = (prediction_column == 0) & (y_target == 1)
		tp = len(predictions[tp_filter])
		tn = len(predictions[tn_filter])
		fp = len(predictions[fp_filter])
		fn = len(predictions[fn_filter])
		true_positive_rate = tp / (tp+fn)
		false_positive_rate = fp / (fp + tn)

	# show different datatypes
	def show_dtypes(self, type):
		df = self.dataframe
		object_columns = df.select_dtypes(include=[type])
		print(object_columns.columns)
		print(object_columns.head(5))

	def change_dtypes(self, columns_to_change):
		pass
		# loans['int_rate'] = loans['int_rate'].str.rstrip('%').astype('float')

	# return x y vars
	def set_features_and_target(self, feature_col, y_target):
		df = self.dataframe
		features = df[feature_col]
		y = df[y_target]
		return features, y_target

	# return x y vars, this is working for logistic regression
	def set_features_and_target1(self,columns, y_target):
		df = self.dataframe
		df = df[columns]
		features = df.drop(y_target, axis=1)
		y_target = df[y_target]
		return features, y_target

	# give x vars as array
	def set_x_y_vars_from_df(self, x_vars, y_target):
		df = self.dataframe
		#df = df_instance
		print(type(df))
		X = df[x_vars].values
		y = df[y_target].values
		return X,y



























"""
columns = ['Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
target = 'USD_BTC_EX_High'
fold = 10
random_state = 3

test_project = ArrangeData(df)
test_project.overall_data_display(1)
test_project.shuffle_rows()
test_project.format_unix_date('date_unix')
test_project.normalize_new_column(columns)
#test_project.time_period_returns('week_highs_avg')
test_project.resample_date('USD_BTC_EX_High', 'month_highs_avg', 'M', 'mean')
test_project.resample_date('USD_BTC_EX_High', 'week_highs_avg', 'W', 'mean')
test_project.resample_date('USD_BTC_EX_High', 'day_highs_avg', 'D', 'mean')
test_project.time_period_returns('week_highs_avg', 'week_highs_avg_change', freq=1)
test_project.time_period_returns('day_highs_avg', '3days_highs_avg_change', freq=3)
test_project.set_binary('week_highs_avg_change', 'target_new', '.05')
test_project.set_mutli_class('3days_highs_avg_change', -.03, 0, .02, .04)
test_project.rolling_average(target, 7)
test_project.rolling_average(target, 14)
test_project.rolling_average(target, 21)
test_project.rolling_average(target, 28)
test_project.set_ones()
test_project.drop_rows(3)
test_project.shuffle_rows()
a = test_project
a.overall_data_display(20)

X = a.dataframe[['ones', 'Transactions_Volume_normalized', 'Number_of_Transactions_normalized', 'LTC_BTC_EX_High_normalized', 'EUR_BTC_EX_High_normalized']].values
y = a.dataframe.target_new.values
"""