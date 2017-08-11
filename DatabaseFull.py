import urllib
import requests
import json
import time
import csv
import sqlite3
import time
import datetime
import calendar
import pandas as pd
import numpy as np

# combining polniex and Database class

class DatabaseFull:

	# vars for class and some are for polniex api
	# opening command sequence for api call
	# this needs to go before every specific command
	opening_command = 'https://poloniex.com/public?command='
	# commands to call to api 
	command_list = ['returnTicker', 'return24hVolume', 'returnTradeHistory&currencyPair=']
	# all possible coins to pull info on
	coin_list = ['BTC_BCN', 'BTC_BELA', 'BTC_BLK', 'BTC_BTCD', 'BTC_BTM', 'BTC_BTS', 'BTC_BURST', 'BTC_CLAM', 'BTC_DASH', 'BTC_DGB', 'BTC_DOGE', 'BTC_EMC2', 'BTC_FLDC', 'BTC_FLO', 'BTC_GAME', 'BTC_GRC', 'BTC_HUC', 'BTC_LTC', 'BTC_MAID', 'BTC_OMNI', 'BTC_NAUT', 'BTC_NAV', 'BTC_NEOS', 'BTC_NMC', 'BTC_NOTE', 'BTC_NXT', 'BTC_PINK', 'BTC_POT', 'BTC_PPC', 'BTC_RIC', 'BTC_SJCX', 'BTC_STR', 'BTC_SYS', 'BTC_VIA', 'BTC_XVC', 'BTC_VRC', 'BTC_VTC', 'BTC_XBC', 'BTC_XCP', 'BTC_XEM', 'BTC_XMR', 'BTC_XPM', 'BTC_XRP', 'USDT_BTC', 'USDT_DASH', 'USDT_LTC', 'USDT_NXT', 'USDT_STR', 'USDT_XMR', 'USDT_XRP', 'XMR_BCN', 'XMR_BLK', 'XMR_BTCD', 'XMR_DASH', 'XMR_LTC', 'XMR_MAID', 'XMR_NXT', 'BTC_ETH', 'USDT_ETH', 'BTC_SC', 'BTC_BCY', 'BTC_EXP', 'BTC_FCT', 'BTC_RADS', 'BTC_AMP', 'BTC_DCR', 'BTC_LSK', 'ETH_LSK', 'BTC_LBC', 'BTC_STEEM', 'ETH_STEEM', 'BTC_SBD', 'BTC_ETC', 'ETH_ETC', 'USDT_ETC', 'BTC_REP', 'USDT_REP', 'ETH_REP', 'BTC_ARDR', 'BTC_ZEC', 'ETH_ZEC', 'USDT_ZEC', 'XMR_ZEC', 'BTC_STRAT', 'BTC_NXC', 'BTC_PASC', 'BTC_GNT', 'ETH_GNT', 'BTC_GNO', 'ETH_GNO']
	# return ticker data provides coins and there data at that moment
	# it returns a dict, below are the keys
	key_values_return_ticker = ['date_time', 'coin_name','id', 'last','lowestAsk','highestBid','percentChange','baseVolume','quoteVolume', 'isFrozen', 'high24hr', 'low24hr']
	# below are the variables return for pair trade history
	# returned is a giant list
	pair_trades_variables = ['globalTradeID','tradeID', 'date', 'type', 'rate', 'amount', 'total']
	date_utc = datetime.datetime.utcnow()
	date_now = datetime.datetime.now()
	# current time right now in unix at UTC
	date_unix_utc = calendar.timegm(date_utc.utctimetuple())
	# current time right now in unix at local time
	date_unix_now = calendar.timegm(date_now.utctimetuple())

	def __init__(self, db_location_base):
		self.db_location_base = db_location_base

	# ticker data to csv
	def current_ticker_to_csv(self, csv_file_name,a_or_w):
		with open(self.db_base_location+csv_file_name, a_or_w) as f:
			writer = csv.DictWriter(f, self.key_values_return_ticker)
			writer.writeheader()
			for coin_number in range((len(coin_list)-1)):
				coin = coin_list[coin_number]
				row_dict = ticker_response_data[coin]
				row_dict_2 = {'date_time':date_unix, 'coin_name':coin}
				row_dict.update(row_dict_2)
				row_dict_all = row_dict
				print(coin)	
				writer.writerow(row_dict)

	# ticker data to sql
	def current_ticker_data_to_sql(self, db_name, table_name):
		ticker_response = requests.get(self.opening_command + self.command_list[0])
		ticker_response_data = ticker_response.json()
		location = self.db_location_base+db_name

		conn=sqlite3.connect(location)
		cur = conn.cursor()
		cur.execute('''CREATE TABLE IF NOT EXISTS  %s
						(human_date_time_utc, date_time, coin_name,id, last,lowestAsk,
						highestBid,percentChange,baseVolume,
						quoteVolume, isFrozen, high24hr, low24hr)'''% (table_name)) 

		for x in range((len(self.coin_list)-1)):
			coin = coin_list[x]
			# info comes back as a dict
			row_dict = ticker_response_data[coin]
			# second dict adding in some more vars
			row_dict_2 = {'human_date_time_uct':self.date_utc, 'date_time':self.date_unix_utc, 'coin_name':coin}
			#combining two dict
			row_dict.update(row_dict_2)
			# adding to sql database
			insert = "INSERT INTO {} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)".format(table_name)
			data_values = [row_dict['human_date_time_uct'], row_dict['date_time'], row_dict['coin_name'], row_dict['id'], row_dict['last'], row_dict['lowestAsk'], row_dict['highestBid'], row_dict['percentChange'], row_dict['baseVolume'], row_dict['quoteVolume'], row_dict['isFrozen'], row_dict['high24hr'], row_dict['low24hr']]
			cur.execute(insert, data_values)
			#cur.execute('INSERT INTO return_ticker_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', [row_dict['date_time'], row_dict['coin_name'], row_dict['id'], row_dict['last'], row_dict['lowestAsk'], row_dict['highestBid'], row_dict['percentChange'], row_dict['baseVolume'], row_dict['quoteVolume'], row_dict['isFrozen'], row_dict['high24hr'], row_dict['low24hr']])
			conn.commit()
		conn.close()

	# convert past trade data from polniex to sql
	def convert_trade_history_to_sql_start_end_vars(self, start_date, end_date, coin_list_array, 
												db_name, coin_name_end):
		conn=sqlite3.connect(self.db_location_base+ db_name)
		cur = conn.cursor()
		for coin in coin_list_array:
			table_name = coin+'_table_'+coin_name_end
			print(table_name)
			cur.execute(''' DROP TABLE IF EXISTS %s''' % (table_name))
			cur.execute('''CREATE TABLE %s 
				(human_date_added, date_time_added_to_db, coin_name, globalTradeID,
				tradeID, date, type, rate, amount, 
				total)''' % (table_name)) 


			# command term
			command_trade_history = self.command_list[2] + coin\
		 + '&start=' + start_date + '&end=' + end_date
			# request using above command term
			return_history_response = requests.get(self.opening_command + command_trade_history)
			# dictionary return of coin pair put in 
			return_history_data = return_history_response.json()
			pair_dict = return_history_data
			#print(coin)
			time.sleep(1)
			#for x in range(3):
			for x in range(len(pair_dict)):
				var0 = self.date_utc
				var1 = self.date_unix_utc
				var2 = coin
				var3 = pair_dict[x]['globalTradeID']
				var4 = pair_dict[x]['tradeID']
				var5 = pair_dict[x]['date']
				var6 = pair_dict[x]['type']
				var7 = pair_dict[x]['rate']
				var8 = pair_dict[x]['amount']
				var9 = pair_dict[x]['total']
				#cur.execute("""INSERT INTO %s VALUES (?,?,?,?,?,?,?,?,?)""", % table_name, (var1,var2,var3,var4,var5,var6,var7,var8,var9))
				insert = "INSERT INTO {} VALUES (?,?,?,?,?,?,?,?,?,?)".format(table_name)
				#print(insert)
				data_values = (var0,var1,var2,var3,var4,var5,var6,var7,var8,var9)
				#cur.execute("""INSERT INTO %s VALUES (?,?,?,?,?,?,?,?,?)""" % table_name (var1,var2,var3,var4,var5,var6,var7,var8,var9))
				cur.execute(insert, data_values)
				#sql
				conn.commit()
		conn.close()

	def aggregate_databases1(self, database_name, table_name_array, columns_wanted_array, time_interval):
		database_dict = {}
		for table in table_name_array:
			con = sqlite3.connect(self.db_location_base+database_name)
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

	def merge_databases_for_models(self, database_name_final, database_dict, **kwargs):
		write_to_db = kwargs.get('write_to_db', None)
		write_to_db_tablename = kwargs.get('write_to_db_tablename', None)
		database_names = list(database_dict.keys())
		print(database_names)
		print(database_names[0])
		combined = database_dict[database_names[0]]
		database_names.pop(0)
		for database_name in database_names:
			combined = combined.merge(database_dict[database_name], how='inner', left_index=True, right_index=True)
		if write_to_db == 'yes':
			con = sqlite3.connect(self.db_location_base+database_name_final)
			combined.to_sql(name=write_to_db_tablename,con=con, if_exists='append')
		return combined


def cycle_over_dates_and_build_coin_db(self, start_period_cycle, end_period_cycle, time_period_interval, limit_interval_before_db_build, coin_list_array, db_name, coin_name_end, database_name, table_name_array, cols_wanted_array, time_interval,write_to_db, write_to_db_tablename):
		start_period_date = datetime.datetime.fromtimestamp(int(start_period_cycle)).strftime('%Y-%m-%d %H:%M:%S')
		end_period_date = datetime.datetime.fromtimestamp(int(end_period_cycle)).strftime('%Y-%m-%d %H:%M:%S')
		wanted_range = pd.date_range(start_period_date, end_period_date, freq=time_period_interval)
		print('wanted range', wanted_range, print(len(wanted_range)))
		array_pair_starts_ends = []
		for x in range(len(wanted_range)):
			array_pair = array_pair = []
			start= wanted_range[x]
			try:
				end= wanted_range[x+1]
			except IndexError:
				end = None
			array_pair.append(start)
			array_pair.append(end)
			array_pair_starts_ends.append(array_pair)
		print('array_pair_starts_ends', array_pair_starts_ends)
		if array_pair_starts_ends[-1][1] == None:
			array_pair_starts_ends.pop(-1)
		print('array_pair_starts_ends2', array_pair_starts_ends)
		split_pairs_in_interval = np.array_split(array_pair_starts_ends, limit_interval_before_db_build)
		print('_______________')
		print('split pairs in interval', split_pairs_in_interval)
		for pairs in split_pairs_in_interval:
			print('pairs')
			print(pairs)
			print('_______________________')
			for x in range(len(pairs)):
				start_date = pairs[x][0]
				end_date = pairs[x][1]
				start_unix = str(int(time.mktime(start_date.timetuple())))
				end_unix = str(int(time.mktime(end_date.timetuple())))
				print('start',start_date, start_unix)
				print('end', end_date, end_unix)
				self.convert_trade_history_to_sql_start_end_vars(start_unix, end_unix,coin_list_array,
															db_name, coin_name_end)
				dbs = self.aggregate_databases1(db_name, table_name_array, cols_wanted_array, 
															time_interval)
				combined_dfs = self.merge_databases_for_models(db_name, dbs, write_to_db=write_to_db,
											write_to_db_tablename=write_to_db_tablename)
		return array_pair_starts_ends, last_value_date, last_value_unix, combined_dfs

