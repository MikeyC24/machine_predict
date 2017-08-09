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

command_list = ['returnTicker', 'return24hVolume', 'returnTradeHistory&currencyPair=']
coin_list = ['BTC_BCN', 'BTC_BELA', 'BTC_BLK', 'BTC_BTCD', 'BTC_BTM', 'BTC_BTS', 'BTC_BURST', 'BTC_CLAM', 'BTC_DASH', 'BTC_DGB', 'BTC_DOGE', 'BTC_EMC2', 'BTC_FLDC', 'BTC_FLO', 'BTC_GAME', 'BTC_GRC', 'BTC_HUC', 'BTC_LTC', 'BTC_MAID', 'BTC_OMNI', 'BTC_NAUT', 'BTC_NAV', 'BTC_NEOS', 'BTC_NMC', 'BTC_NOTE', 'BTC_NXT', 'BTC_PINK', 'BTC_POT', 'BTC_PPC', 'BTC_RIC', 'BTC_SJCX', 'BTC_STR', 'BTC_SYS', 'BTC_VIA', 'BTC_XVC', 'BTC_VRC', 'BTC_VTC', 'BTC_XBC', 'BTC_XCP', 'BTC_XEM', 'BTC_XMR', 'BTC_XPM', 'BTC_XRP', 'USDT_BTC', 'USDT_DASH', 'USDT_LTC', 'USDT_NXT', 'USDT_STR', 'USDT_XMR', 'USDT_XRP', 'XMR_BCN', 'XMR_BLK', 'XMR_BTCD', 'XMR_DASH', 'XMR_LTC', 'XMR_MAID', 'XMR_NXT', 'BTC_ETH', 'USDT_ETH', 'BTC_SC', 'BTC_BCY', 'BTC_EXP', 'BTC_FCT', 'BTC_RADS', 'BTC_AMP', 'BTC_DCR', 'BTC_LSK', 'ETH_LSK', 'BTC_LBC', 'BTC_STEEM', 'ETH_STEEM', 'BTC_SBD', 'BTC_ETC', 'ETH_ETC', 'USDT_ETC', 'BTC_REP', 'USDT_REP', 'ETH_REP', 'BTC_ARDR', 'BTC_ZEC', 'ETH_ZEC', 'USDT_ZEC', 'XMR_ZEC', 'BTC_STRAT', 'BTC_NXC', 'BTC_PASC', 'BTC_GNT', 'ETH_GNT', 'BTC_GNO', 'ETH_GNO']
key_values_return_ticker = ['date_time', 'coin_name','id', 'last','lowestAsk','highestBid','percentChange','baseVolume','quoteVolume', 'isFrozen', 'high24hr', 'low24hr']
pair_trades_key_list1 = ['globalTradeID','tradeID', 'date', 'type', 'rate', 'amount', 'total']
command_ticker_now = command_list[0]
date_utc = datetime.datetime.utcnow()
date_now = datetime.datetime.now()
date_unix_utc = calendar.timegm(date_utc.utctimetuple())
date_unix_now = calendar.timegm(date_now.utctimetuple())
#print(date_unix_utc)
#print(date_unix_now)
#print(date_now)

# enter date and time here to get its unix, needed to send to api
# answer will be in UTC zone
# enter each var indivdually
def unix_converter2(y, mo, d, h, mi, s):
	full_date = datetime.datetime(y,mo,d,h,mi,s)
	unix = calendar.timegm(full_date.utctimetuple())
	return unix

# delta measure options are seconds, minutes, hours, days
def time_delta(y, mo, d, h, mi, s,delta_measure, delta_amount):
	full_date = datetime.datetime(y,mo,d,h,mi,s)
	if delta_measure == 'days':
		delta = datetime.timedelta(days = delta_amount)
	elif delta_measure == 'hours':
		delta = datetime.timedelta(hours = delta_amount)
	elif delta_measure == 'minutes':
		delta = datetime.timedelta(minutes = delta_amount)
	elif delta_measure == 'seconds':
		delta = datetime.timedelta(seconds= delta_amount)
	else:
		return 'not an option for delta measure'
	new_date = full_date - delta
	unix_new_date = calendar.timegm(new_date.utctimetuple())
	return unix_new_date

# convert unix to human readable date
def convert_unix_to_date(unix):
	date_read = datetime.datetime.fromtimestamp(int(unix)).strftime('%Y-%m-%d %H:%M:%S')
	return(date_read)

class PolniexApiData:
	# class variables
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

	def __init__(self, start_date, end_date, location_base):
		self.start_date = start_date
		self.end_date = end_date
		self.location_base = location_base

	# ticker data to csv
	def current_ticker_to_csv(self, csv_file_name,a_or_w):
		with open(self.base_location+csv_file_name, a_or_w) as f:
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

	def current_ticker_data_to_sql(self, db_name, table_name):
		ticker_response = requests.get(self.opening_command + self.command_list[0])
		ticker_response_data = ticker_response.json()
		location = self.location_base+db_name

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

	def convert_trade_history_to_sql(self, coin_list_array, db_name, coin_name_end):
		conn=sqlite3.connect(location_base+ db_name)
		cur = conn.cursor()
		for coin in coin_list_array:
			table_name = coin+'_table_'+coin_name_end
			print(table_name)
			cur.execute('''CREATE TABLE IF NOT EXISTS 
				%s 
				(human_date_added, date_time_added_to_db, coin_name, globalTradeID,
				tradeID, date, type, rate, amount, 
				total)''' % (table_name)) 


			# command term
			command_trade_history = self.command_list[2] + coin\
		 + '&start=' + self.start_date + '&end=' + self.end_date
			# request using above command term
			return_history_response = requests.get(self.opening_command + command_trade_history)
			# dictionary return of coin pair put in 
			return_history_data = return_history_response.json()
			pair_dict = return_history_data
			#print(coin)
			time.sleep(1)
			#for x in range(3):
			for x in range(len(pair_dict)):
				var0 = date_utc
				var1 = date_unix_utc
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


		# start and end are in unix
	def cycle_over_dates_and_build_coin_db(self, start_period_cycle, end_period_cycle, time_period_interval, limit_interval_before_db_build):
		start_period_date = datetime.datetime.fromtimestamp(int(start_period_cycle)).strftime('%Y-%m-%d %H:%M:%S')
		end_period_date = datetime.datetime.fromtimestamp(int(end_period_cycle)).strftime('%Y-%m-%d %H:%M:%S')
		wanted_range = pd.date_range(start_period_date, end_period_date, freq=time_period_interval)
		array_pair_starts_ends = []
		#while len(array_pair_starts_ends) < limit_interval_before_db_build:
		#array_pair_starts_ends = []
		for x in range(len(wanted_range)):
			array_pair = []
			start= wanted_range[x]
			try:
				end= wanted_range[x+1]
			except IndexError:
				end = None
			array_pair.append(start)
			array_pair.append(end)
			array_pair_starts_ends.append(array_pair)
		#return array_pair_starts_ends
		if array_pair_starts_ends[-1][1] == None:
			# next start value
			pick_up_value = array_pair_starts_ends[-1].pop(0)
			array_pair_starts_ends.pop(-1)
		else:
			pick_up_value = array_pair_starts_ends[-1][1]
		#print(array_pair_starts_ends)
		#print(pick_up_value)	
		return array_pair_starts_ends, pick_up_value

# '/home/mike/Documents/coding_all/data_sets_machine_predict/3_coin_test_db'
location_base = '/home/mike/Documents/coding_all/data_sets_machine_predict/'
# 8/7/17 100 GMT
start_date =  '1502067600'
# 8/7/17 1300 GMT
end_date = '1502110800'
database_name = '3_coin_test_db'
# start date weekend
# saturday 8/5/17 at 1200am GMT
start_wke = '1501891200'
#end weekend
# monday 8/7/17 1am GMT
end_wke = '1502067600'
time_interval_delta_measure = 'h'
time_interval_delta_amount = 3


data_class = PolniexApiData(start_date,end_date,location_base)
#data_class.current_ticker_data_to_sql('new_class_test_db1', 'cool_table_name')
coin_list_test = ['BTC_LTC', 'BTC_ETH']
coin_list_cont = [ 'ETH_LSK', 'BTC_LBC', 'BTC_STEEM', 'ETH_STEEM', 'BTC_SBD', 'BTC_ETC', 'ETH_ETC', 'USDT_ETC', 'BTC_REP', 'USDT_REP', 'ETH_REP', 'BTC_ARDR', 'BTC_ZEC', 'ETH_ZEC', 'USDT_ZEC', 'XMR_ZEC', 'BTC_STRAT', 'BTC_NXC', 'BTC_PASC', 'BTC_GNT', 'ETH_GNT', 'BTC_GNO', 'ETH_GNO']
top_3_coin_list = ['USDT_ETH', 'USDT_BTC', 'USDT_LTC']
#data_class.convert_trade_history_to_sql(top_3_coin_list, database_name, '')
# '/home/mike/Documents/coding_all/data_sets_machine_predict/all_coin_history_db_big
# all_coin_history_db_big


results = data_class.cycle_over_dates_and_build_coin_db(start_wke, end_wke, 'H', 3)
print('aray', results[0])
print('picp up', results[1])
"""
for result in results:
	print(result)
	print('___________')
print(type(results))
print(len(results))
print(results[0])
print(results[0][0])
print(results[0][1])
"""











"""
# start and end are in unix
def cycle_over_dates_and_build_coin_db(start_period, end_period, time_period_interval, limit_interval_before_db_build):
	start_period_date = datetime.datetime.fromtimestamp(int(start_period)).strftime('%Y-%m-%d %H:%M:%S')
	end_period_date = datetime.datetime.fromtimestamp(int(end_period)).strftime('%Y-%m-%d %H:%M:%S')
	wanted_range = pd.date_range(start_period_date, end_period_date, freq=time_period_interval)
	array_pair_starts_ends = []
	for x in range(len(wanted_range)):
		array_pair = []
		start= wanted_range[x]
		try:
			end= wanted_range[x+1]
		except IndexError:
			end = 'no more values'
		array_pair.append(start)
		array_pair.append(end)
		print('array_pair', array_pair)
		array_pair_starts_ends.append(array_pair)
	return array_pair_starts_ends

results = cycle_over_dates_and_build_coin_db(start_wke, end_wke, 'H', 3)
for result in results:
	print(result)
	print('___________')
print(type(results))
print(len(results))
print(results[0])
print(results[0][0])
print(results[0][1])

def cycle_dates_to_pull_polinex_api(dates_list):
	pass
"""