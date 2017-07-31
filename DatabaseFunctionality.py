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

	def __init__(self, metric_dictionary, location_base):
		self.metric_dictionary = metric_dictionary
		self.location_base = location_base

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




