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

	def drop_certain_percent_of_missing_data(self, percent):
		df = self.dataframe
		count = len(df)*percent
		print(count)
		df.dropna(thresh=count, axis=1, inplace=True)
		return df

	def drop_columns_array(self, column_array):
		df = self.dataframe
		df.drop(column_array, axis = 1, inplace=True)
		return df

	def map_target_for_binary(self, target, yes, no):
		df =self.dataframe
		df = df[(df[target] == yes) | (df[target] == no)]
		dict_map = {target: {yes:1, no:2}}
		df = df.replace(dict_map)
		return df

	def drop_nan_values(self):
		df = self.dataframe
		df.dropna(inplace=True)
		return df

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
			