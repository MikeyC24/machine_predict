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