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
		self.dataframe.dropna(thresh=count, axis=1, inplace=True)
		return self

	# takes in a array of columns to drop from data
	# cat = remove
	def drop_columns_array(self, column_array):
		df = self.dataframe
		df.drop(column_array, axis = 1, inplace=True)
		return df

	# user inputs target feature and what they want for the 1 and o 
	# value then drops everything else in column
	#cat = change
	def map_target_for_binary(self, target, yes, no):
		df =self.dataframe
		df = df[(df[target] == yes) | (df[target] == no)]
		dict_map = {target: {yes:1, no:2}}
		df = df.replace(dict_map)
		return df

	# drop all missing data leaving a dataframe with no missing data
	# CAT = REMOVE
	def drop_nan_values(self):
		df = self.dataframe
		df.dropna(inplace=True)
		return df

	# drops all columns with only one unique value
	# CAT = remove
	def drop_cols_with_one_unique_value(self):
		df = self.dataframe
		cols_to_drop = []
		for x in df:
			non_null = df[x].dropna()
			unique_non_null = non_null.unique()
			num_true_unique = len(unique_non_null)
			cols_to_drop.append(x) if num_true_unique == 1 else False
		df.drop(cols_to_drop, axis=1, inplace=True)
		return df

	# show count of each columns unique values 
	# this can return self or a df that is entered this is done
	# bc if dataframe has been reassigned it needs to take in the new df
	# cat = show
	def show_unique_count_each_col(self, dataframe=None):
		df = self.dataframe
		if dataframe is not None:
			df = df1
		for x in list(df):
			print(x, len(df[x].unique()))

	# show the number of nan value sin each column
	# this can return self or a df that is entered this is done
	# bc if dataframe has been reassigned it needs to take in the new df
	# cat = show
	def show_nan_count(self, percent = 0, dataframe_new=None):
		df = self.dataframe
		if dataframe_new is not None:
			df = dataframe_new
		len_col = df.shape[0]
		data_dict = {}
		for x in list(df):
			missing_col_values = df[x].isnull().sum()
			if missing_col_values > (len_col * percent):
				data_dict[str(x)] = missing_col_values
		for k,v in data_dict.items():
			print(str(k)+':'+str(v))

	# drop column by percent missing 
	# cat = remove
	def remove_col_by_percent_missing(self, percent):
		df = self.dataframe
		len_col = df.shape[0]
		one_percent_missing = []
		for x in list(df):
			missing_col_values = df[x].isnull().sum()
			if missing_col_values > (len_col * percent):
				one_percent_missing.append(x)
		df = df.drop(one_percent_missing, axis = 1)
		return df

	# show the dtype of each column
	# cat = show
	def show_all_dtypes(self, dataframe_new = None):
		df = self.dataframe
		if dataframe_new is not None:
			df = dataframe_new
		for x in list(df):
			print(x, df[x].dtype)







			