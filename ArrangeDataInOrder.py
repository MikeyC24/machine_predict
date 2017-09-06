import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import datetime
import time
from itertools import cycle
import math

class ArrangeDataInOrder:

	# this class takes in a data frame and manipulates it
	def __init__(self, dataframe):
		self.dataframe = dataframe

	# this si for time series, this is to fill in missing time range gaps, 
	# can set index and extroplate missing data
	# needs to be done separate from other methods
	# category = pre data manipulation time series
	def fill_in_data_full_range(self, start_date, end_date, freq, index='no',
							interpolate='yes'):
		if index != 'no':
			df = self.dataframe.set_index(index)
		else:
			df = self.dataframe
		#print(df.head(10))
		# make datetime
		df.index = pd.DatetimeIndex(df.index)
		# get rid of dups
		df = df.groupby(df.index).first()
		# set new range 
		drange = pd.date_range(start=start_date, end=end_date, freq=freq)
		df = df.reindex(drange)
		if interpolate == 'yes':
			df = df.interpolate()
		return df

	# this is for time series, can reset data by new group
	# needs to be done separate than other methods
	# category = pre data manipulation time series
	def group_by_time_with_vars(self, time_interval, reset_index='yes', interpolate='no', index='no',
								set_to_datetime='no'):
		df =self.dataframe
		if set_to_datetime != 'no':
			df['date'] = pd.to_datetime(df['date'])
		#print('df after datetime', df)
		#df.index = df['date']
		if index != 'no':
			df.set_index(['date'], inplace=True, drop=False)
		if interpolate == 'yes':
			df = df.resample(time_interval).mean().interpolate()
		else:
			df = df.resample(time_interval).mean()
		df.reset_index(inplace=True) if reset_index == 'yes' else print('no reset index')
		#print('df inside broup by', df.head(20))
		return df

	# method to drop columns if a certain amount of data is missing
	# cat = remove
	def drop_certain_percent_of_missing_data(self, percent):
		#df = self.dataframe
		count = len(self.dataframe)*percent
		print(count)
		self.dataframe.dropna(thresh=count, axis=1, inplace=True)
		return self

	# takes in a array of columns to drop from data
	# cat = remove
	def drop_columns_array(self, column_array):
		self.dataframe.drop(column_array, axis = 1, inplace=True)
		return self

	# user inputs target feature and what they want for the 1 and o 
	# value then drops everything else in column
	#cat = change
	def map_target_for_binary(self, target, yes, no):
		df =self.dataframe
		self.dataframe = self.dataframe[(self.dataframe[target] == yes) | (self.dataframe[target] == no)]
		dict_map = {target: {yes:1, no:2}}
		self.dataframe = self.dataframe.replace(dict_map)
		return self

	# drop all missing data leaving a dataframe with no missing data
	# CAT = REMOVE
	def drop_nan_values(self):
		self.dataframe.dropna(inplace=True)
		return self

	# drops all columns with only one unique value
	# CAT = remove
	def drop_cols_with_one_unique_value(self):
		cols_to_drop = []
		for x in self.dataframe:
			non_null = self.dataframe[x].dropna()
			unique_non_null = non_null.unique()
			num_true_unique = len(unique_non_null)
			cols_to_drop.append(x) if num_true_unique == 1 else False
		self.dataframe.drop(cols_to_drop, axis=1, inplace=True)
		return self

	# show count of each columns unique values 
	# cat = show
	def show_unique_count_each_col(self):
		for x in list(self.dataframe):
			print(x, len(self.dataframe[x].unique()))

	# show the number of nan value sin each column
	# cat = show
	def show_nan_count(self, percent = 0):
		len_col = self.dataframe.shape[0]
		data_dict = {}
		for x in list(self.dataframe):
			missing_col_values = self.dataframe[x].isnull().sum()
			if missing_col_values > (len_col * percent):
				data_dict[str(x)] = missing_col_values
		for k,v in data_dict.items():
			print(str(k)+':'+str(v))

	# drop column by percent missing 
	# cat = remove
	def remove_col_by_percent_missing(self, percent):
		len_col = self.dataframe.shape[0]
		one_percent_missing = []
		for x in list(self.dataframe):
			missing_col_values = self.dataframe[x].isnull().sum()
			if missing_col_values > (len_col * percent):
				one_percent_missing.append(x)
		self.dataframe.drop(one_percent_missing, axis = 1,inplace=True)
		return self

	# show the dtype of each column
	# cat = show
	def show_all_dtypes(self, type='all'):
		if type == 'all':
			for x in list(self.dataframe):
				print(x, self.dataframe[x].dtype)
		else:
			object_columns_df = self.dataframe.select_dtypes(include=[type])
			print(object_columns_df)

	def convert_to_numerical(self, cols_array, type='default'):
		if type == 'default':
			for col in cols_array:
				self.dataframe[col] = self.dataframe[col].astype('float')
		elif type == 'percent':
			for col in cols_array:
				self.dataframe[col] = self.dataframe[col].str.rstrip('%').astype('float')
		else:
			print('that type is not recongized')
		return self

	def use_dict_map_for_new_cols(self, dict_map):
		self.dataframe = self.dataframe.replace(dict_map)
		return self

	def convert_cols_to_dummies(self, col_array):
		for col in col_array:
			self.dataframe[col] = self.dataframe[col].astype('category')
		dummy_df = pd.get_dummies(self.dataframe[col_array])
		self.dataframe = pd.concat([self.dataframe, dummy_df], axis=1)
		self.dataframe = self.dataframe.drop(col_array, axis=1)
		return self





			