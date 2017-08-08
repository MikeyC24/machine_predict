import urllib
import requests
import json
import time
import csv
import sqlite3
import time
import datetime
#from datetime import datetime
import calendar
import pandas as pd
import numpy as np

# 7/31/17
# goal of this class, as of now is two fold
# 1. write results dict to a data base with needed data and time they were added
# be able to take from multple databases, combine the needed info and feed into models

# keys for user params - in layers
# layer one - dict_results_simple, dict_results_kfold, dict_results_train_set
# layer two - LogisticRegress, DecisionTreeCla, MLPClassifier
# layer three - sensitivity': , 'false_negative_rate', 'fallout_rate', 'specificty', 'precision', 'false_discovery_rate', 'negative_predictive_value', 'roc_auc_score', 'mse', 'mae', 'r2_score', 'variance', 'ACC'
# layer four = the scores

class DatabaseFunctionality:

	def __init__(self, db_location_base, database_name):
		self.db_location_base = db_location_base
		self.database_name = database_name

	"""
	# this is being done in the machineprdictmodel class
	def write_results_to_db_for_user_params(self, database_name table_name):
		location = self.location_base+data_base_name
		date_utc = datetime.datetime.utcnow()
		# open and con to db
		conn = sqlite3.connect(location)
		curr = conn.cursor()
		cur.execute('''CREATE TABLE IF NOT EXISTS %s
					(date_added, train_type, model_type, score_name)
					''') % (table_name)
		for keys, values in self.metric_dictionary.items():
			if len(values) > 0:
				add
			else:
				print('all traing method dicts are empty')
	"""
	"""
	# columns in table 0|human_date_added||0||0
1|date_time_added_to_db||0||0 2|coin_name||0||0 3|globalTradeID||0||0
4|tradeID||0||0 5|date||0||0 6|type||0||0 7|rate||0||0
8|amount||0||0 9|total||0||0
	"""
	def aggregate_databases(self, table_name_array, columns_wanted_array):
		con = sqlite3.connect(self.db_location_base+self.database_name)
		cur = con.cursor()
		df1 = pd.read_sql_query('SELECT * FROM %s' % (table_name_array[0]), con)
		df2 = pd.read_sql_query('SELECT * FROM %s' % (table_name_array), con)
		key = columns_wanted_array[0]
		#could make all columns wanted into a series and concat that way 
		# then reset index by ignore_index = True
		# figure it out from here = http://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html

	def aggregate_databases1(self, table_name_array, columns_wanted_array, time_interval):
		database_dict = {}
		for table in table_name_array:
			con = sqlite3.connect(self.db_location_base+self.database_name)
			cur = con.cursor()
			df = pd.read_sql_query('SELECT * FROM %s' % (table), con)
			df = df.loc[:, columns_wanted_array]
			column_names = cur.description
			coin = df['coin_name'][0]

			df.columns = df.columns + '_'  + coin
			date_col = 'date_'+coin
			df['date'] = df[date_col]
			total_col = 'total_' + coin
			cols_num = ['rate', 'amount', 'total']
			cols_num1 = []
			for x in cols_num:
				new_name = x+'_'+coin
				cols_num1.append(new_name)
			for col in cols_num1:
				df[col] = pd.to_numeric(df[col], errors='coerce')
			#df.index = df[date_col]
			df['date'] = pd.to_datetime(df['date'])
			df.index = df['date']
			df1 = df.groupby(pd.TimeGrouper(time_interval)).mean() 
			df2 = df.groupby(pd.TimeGrouper(time_interval)).count()
			total_name = 'total'+'_'+coin
			freq_series = df2[total_name]
			df1['trade_count_'+coin] = freq_series
			df1['trade_count_'+coin] = pd.to_numeric(df1['trade_count_'+coin], errors='coerce')
			df = df1
			# https://chrisalbon.com/python/pandas_apply_operations_to_groups.html
			database_dict[str(coin) + 'formatted'] = df
		return database_dict

	def merge_databases_for_models(self, database_dict, **kwargs):
		write_to_db = kwargs.get('write_to_db', None)
		write_to_db_tablename = kwargs.get('write_to_db_tablename', None)
		database_names = list(database_dict.keys())
		print(database_names)
		print(database_names[0])
		combined = database_dict[database_names[0]]
		database_names.pop(0)
		"""
		print(database_names)
		print('combined', combined.head(5))
		db2 = database_dict[database_names[0]]
		print('db2', db2.head(5))
		combined = combined.merge(db2, how='inner', left_index=True, right_index=True)
		print('combined2', combined.head(5))
		"""
		for database_name in database_names:
			combined = combined.merge(database_dict[database_name], how='inner', left_index=True, right_index=True)
		#combined = pd.concat(database_dict.values())
		if write_to_db == 'yes':
			con = sqlite3.connect(self.db_location_base+self.database_name)
			combined.to_sql(name=write_to_db_tablename,con=con, if_exists='append')
		return combined



"""
saving in case
	def aggregate_databases1(self, table_name_array, columns_wanted_array):
		database_dict = {}
		for table in table_name_array:
			con = sqlite3.connect(self.db_location_base+self.database_name)
			cur = con.cursor()
			df = pd.read_sql_query('SELECT * FROM %s' % (table), con)
			df = df.loc[:, columns_wanted_array]
			column_names = cur.description
			coin = df['coin_name'][0]
			
			df.columns = df.columns + '_'  + coin
			date_col = 'date_'+coin
			total_col = 'total_' + coin
			cols_num = ['rate', 'amount', 'total']
			cols_num1 = []
			for x in cols_num:
				new_name = x+'_'+coin
				cols_num1.append(new_name)
			for col in cols_num1:
				df[col] = pd.to_numeric(df[col], errors='coerce')
			#df.index = df[date_col]
			df[date_col] = pd.to_datetime(df[date_col])
			df.index = df[date_col]
			df1 = df.groupby(pd.TimeGrouper('10Min')).mean()
			df2 = df.groupby(pd.TimeGrouper('10Min')).count()
			total_name = 'total'+'_'+coin
			freq_series = df2[total_name]
			df1['trade_count_'+coin] = freq_series
			df = df1
			# https://chrisalbon.com/python/pandas_apply_operations_to_groups.html
			database_dict[str(coin) + 'formatted'] = df
		return database_dict
"""