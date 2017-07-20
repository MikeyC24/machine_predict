"""

def time_period_returns_array(dict_vars):
	#df = self.dataframe
	#for keys, values in dict_vars.items():
	#range_count = int(len(list(dict_vars.values()))/len(list(dict_vars.keys())))
	#print(len(list(dict_vars.values())))
	#print(range_count)
	#range_count = range_count+1
	#print(range_count)
	count = sum(len(v) for v in dict_varss1.values())
	range_count = int(count / len(list(dict_vars.keys())))
	print(count)
	for x in range(range_count):
		#print(keys, values)
		print(x)
		print(dict_vars['column_name_old'][x])
		print(dict_vars['column_name_new'][x])
		print(dict_vars['freq'][x])
		#prices = df[column_name_old]
		#print(type(prices))
		#df[column_name_new] = prices.pct_change(freq)
	#return df


dict_varss = {'column_name_old': 'colold', 'column_name_new':'colnew', 'freq':1, 'column_name_old': 'colold1', 'column_name_new':'colnew1', 'freq':3}
dict_varss1 = {'column_name_old':['colold', '2ndcol'], 'column_name_new':['colnew', '2ndcol'], 'freq':[1,2]}
time_period_returns_array(dict_varss1)
count = sum(len(v) for v in dict_varss1.values())
#print(count)
"""
"""
print(len(dict_varss))
#count = sum(len(v) for v in dict_varss.values())
#print(count)
countv = list(dict_varss.keys())
print(countv)
print(len(countv))
print(len(countv)/3)
#time_period_returns_array(dict_varss)
"""
"""
param_dict = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':None, 'random_state':None, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
print(param_dict['penalty'])
print(type(param_dict['penalty']))
for x,y in param_dict.items():
	print(x,y)

"""
#dict1 = {'param_dict_vars': {'penalty': 'l2', 'dual': False, 'tol': 0.0001, 'C': 1.0, 'fit_intercept': True, 'intercept_scaling': 1, 'class_weight': 'balanced', 'random_state': 1, 'solver': 'liblinear', 'max_iter': 100, 'multi_class': 'ovr', 'verbose': 0, 'warm_start': False, 'n_jobs': 1}}
#print(dict1['param_dict_vars']['penalty'])
"""
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

current_utc_time = '1500466815'
print(current_utc_time.datetime.date)
"""
"""
array1 = [1, 'test', {'hi':1, 'sample':'output'}]
for x in array1:
	print(x)
y = array1[2]
print(type(y))
for x,z in y.items():
	print(x,z)
print(y['hi'])

"""
model_score_dict = {'logistic':{'error_metric':'roc_auc_score', 'tpr_range'[.06,1], 'fpr_range'[.0,.05]}}
for x,y in model_score_dict.items():
	print(x,y)