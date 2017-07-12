import urllib
import requests
import json
import time
import csv
import sqlite3
import time
import datetime
from datetime import datetime
import calendar


command_list = ['returnTicker', 'return24hVolume', 'returnTradeHistory&currencyPair=']
coin_list = ['BTC_BCN', 'BTC_BELA', 'BTC_BLK', 'BTC_BTCD', 'BTC_BTM', 'BTC_BTS', 'BTC_BURST', 'BTC_CLAM', 'BTC_DASH', 'BTC_DGB', 'BTC_DOGE', 'BTC_EMC2', 'BTC_FLDC', 'BTC_FLO', 'BTC_GAME', 'BTC_GRC', 'BTC_HUC', 'BTC_LTC', 'BTC_MAID', 'BTC_OMNI', 'BTC_NAUT', 'BTC_NAV', 'BTC_NEOS', 'BTC_NMC', 'BTC_NOTE', 'BTC_NXT', 'BTC_PINK', 'BTC_POT', 'BTC_PPC', 'BTC_RIC', 'BTC_SJCX', 'BTC_STR', 'BTC_SYS', 'BTC_VIA', 'BTC_XVC', 'BTC_VRC', 'BTC_VTC', 'BTC_XBC', 'BTC_XCP', 'BTC_XEM', 'BTC_XMR', 'BTC_XPM', 'BTC_XRP', 'USDT_BTC', 'USDT_DASH', 'USDT_LTC', 'USDT_NXT', 'USDT_STR', 'USDT_XMR', 'USDT_XRP', 'XMR_BCN', 'XMR_BLK', 'XMR_BTCD', 'XMR_DASH', 'XMR_LTC', 'XMR_MAID', 'XMR_NXT', 'BTC_ETH', 'USDT_ETH', 'BTC_SC', 'BTC_BCY', 'BTC_EXP', 'BTC_FCT', 'BTC_RADS', 'BTC_AMP', 'BTC_DCR', 'BTC_LSK', 'ETH_LSK', 'BTC_LBC', 'BTC_STEEM', 'ETH_STEEM', 'BTC_SBD', 'BTC_ETC', 'ETH_ETC', 'USDT_ETC', 'BTC_REP', 'USDT_REP', 'ETH_REP', 'BTC_ARDR', 'BTC_ZEC', 'ETH_ZEC', 'USDT_ZEC', 'XMR_ZEC', 'BTC_STRAT', 'BTC_NXC', 'BTC_PASC', 'BTC_GNT', 'ETH_GNT', 'BTC_GNO', 'ETH_GNO']
# returns current market snapshot of highest bids, asks, percent change and volume
# can we do this on a 10 minute time basis (144 hits a day), if not hourly
## time method

# append ticker_response_data
# returns a dict where each key is the coin
# in the dict has a dict with the column type and value
#print(ticker_response_data)
#print(ticker_response_data['ETH_GNO']['id'])
print((len(coin_list)-1))

key_values_return_ticker = ['date_time', 'coin_name','id', 'last','lowestAsk','highestBid','percentChange','baseVolume','quoteVolume', 'isFrozen', 'high24hr', 'low24hr']
command_ticker_now = command_list[0]
ticker_response = requests.get('https://poloniex.com/public?command=' + command_ticker_now)
ticker_response_data = ticker_response.json()
d = datetime.utcnow()
date_unix = calendar.timegm(d.utctimetuple())

"""
# this is working and will store to csv
with open('/home/mike/Documents/coding_all/machine_learn/machine_predict/ticker_response_data.csv', 'a') as f:
	writer = csv.DictWriter(f, key_values_return_ticker)
	writer.writeheader()
	for coin_number in range((len(coin_list)-1)):
		coin = coin_list[coin_number]
		row_dict = ticker_response_data[coin]
		row_dict_2 = {'date_time':date_unix, 'coin_name':coin}
		row_dict.update(row_dict_2)
		row_dict_all = row_dict
		print(coin)	
		writer.writerow(row_dict)
"""
"""
# /home/mike/Documents/coding_all/machine_learn/machine_predict/return_ticker_db
# this is working for sql db but needs to be updated to append new data
location_base = '/home/mike/Documents/coding_all/machine_learn/machine_predict/'
conn=sqlite3.connect(location_base+'return_ticker_db')
cur = conn.cursor()
cur.execute('''CREATE TABLE return_ticker_data
				(date_time, coin_name,id, last,lowestAsk,
				highestBid,percentChange,baseVolume,
				quoteVolume, isFrozen, high24hr, low24hr)''')

for x in range((len(coin_list)-1)):
	coin = coin_list[x]
	row_dict = ticker_response_data[coin]
	row_dict_2 = {'date_time':date_unix, 'coin_name':coin}
	row_dict.update(row_dict_2)
	cur.execute('INSERT INTO return_ticker_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', [row_dict['date_time'], row_dict['coin_name'], row_dict['id'], row_dict['last'], row_dict['lowestAsk'], row_dict['highestBid'], row_dict['percentChange'], row_dict['baseVolume'], row_dict['quoteVolume'], row_dict['isFrozen'], row_dict['high24hr'], row_dict['low24hr']])
	conn.commit()
conn.close()
"""


#past history 
# csv
# at first we need to pull as much as we can, set target a year back,will limit us at 50k
#inital history 
pair_trades_key_list = ['date_time_added_to_db', 'coin_name','globalTradeID','tradeID', 'date', 'type', 'rate', 'amount', 'total']
pair_trades_key_list1 = ['globalTradeID','tradeID', 'date', 'type', 'rate', 'amount', 'total']
# this unix time stamp is two years back
start_date =  '1484174782'
start_date1 = '1410158341'
# use this site to get most recent one  https://www.epochconverter.com/
end_date = '1499813092'
coin_list_test = ['BTC_ETH', 'BTC_NXT']
# this is writing to csv but csv file contains too much data too open
# check code at bottom for structure, this is a dict in an array
# lots if iterating to do
"""
with open('/home/mike/Documents/coding_all/machine_predict/pair_trading_history_data.csv', 'w') as f:
	writer = csv.writer(f,pair_trades_key_list1)
	for coin in coin_list_test:
		# command term
		command_trade_history = command_trade_history = command_list[2] + coin\
	 + '&start=' + start_date + '&end=' + end_date
		# request using above command term
		return_history_response = requests.get('https://poloniex.com/public?command=' + command_trade_history)
		# dictionary return of coin pair put in 
		return_history_data = return_history_response.json()
		pair_dict = return_history_data
		for x in range(2):
			trade = pair_dict[0]
			for key in pair_trades_key_list1:
				#print(coin,x,key,pair_dict[x][key])
				value = pair_dict[x][key]
				trade_dict = {key:value}
				#trade_dict2 = {'coin_name':coin}
				#trade_dict.update(trade_dict2)
				#print(trade_dict)
				#pair_dict2 = {'date_time_added_to_db':date_unix, 'coin_name':coin}
				#pair_dict.update(pair_dict2)
				#print(coin, pair_dict)
				writer.writerow([key,value])
				#time.sleep(1)
	# append return_history_data
# then we need to store to database
# this returns a dict with the below keys
# ['globalTradeID','tradeID', 'date', 'type', 'rate', 'amount', 'total']
# for this we need to return the above as columns along with the values and 
# make sure it is tied to its coin, coin from the iteration

# after we get as much as we can
# then we need this ti run a on some time interval changing the unix date
# start and stop and appending to database
"""
"""
# sql for par trade history 
location_base = '/home/mike/Documents/coding_all/machine_predict/'
conn=sqlite3.connect(location_base+'pair_trade_history_db')
cur = conn.cursor()
cur.execute('''CREATE TABLE pair_trade_history_db
				(date_time_added_to_db, coin_name, globalTradeID,
				tradeID, date, type, rate, amount, total)''')

for coin in coin_list_test:
		# command term
		command_trade_history = command_trade_history = command_list[2] + coin\
	 + '&start=' + start_date + '&end=' + end_date
		# request using above command term
		return_history_response = requests.get('https://poloniex.com/public?command=' + command_trade_history)
		# dictionary return of coin pair put in 
		return_history_data = return_history_response.json()
		pair_dict = return_history_data
		for x in range(2):
			trade = pair_dict[0]
			for key in pair_trades_key_list1:
				#print(coin,x,key,pair_dict[x][key])
				value = pair_dict[x][key]
				pair_dict1 = {key:value}
				trade_dict2 = {'coin_name':coin}
				pair_dict1.update(trade_dict2)
				print(pair_dict1)
				#pair_dict2 = {'date_time_added_to_db':date_unix, 'coin_name':coin}
				#pair_dict.update(pair_dict2)
				#print(coin, pair_dict)
				#time.sleep(1)
				cur.execute('INSERT INTO pair_trade_history_db VALUES (?,?,?,?,?,?,?,?)', ([pair_dict1['coin_name'], pair_dict1['globalTradeID'], pair_dict1['tradeID'], pair_dict1['date'], pair_dict1['type'], pair_dict1['rate'], pair_dict1['amount'], pair_dict1['total']]))
				conn.commit()

conn.close()
"""
# [pair_dict['date_time_added_to_db'], pair_dict['coin_name'], pair_dict['globalTradeID'], pair_dict['tradeID'], pair_dict['date'], pair_dict['type'], pair_dict['rate'], pair_dict['amount'], pair_dict['total']])
"""
start_date2 = '1484174782'
for coin in coin_list_test:	
	print(coin)
	# command term
	command_trade_history = command_trade_history = command_list[2] + coin\
	 + '&start=' + start_date2 + '&end=' + end_date
	#print(command_trade_history)
	#print(start_date)
	#print(end_date)
	#print(type(start_date))
	#print(type(end_date))
	print('https://poloniex.com/public?command=' + command_trade_history)
	# request using above command term
	return_history_response = requests.get('https://poloniex.com/public?command=' + command_trade_history)
	# dictionary return of coin pair put in 
	return_history_data = return_history_response.json()
	print(return_history_data)
"""

"""
test = 'https://poloniex.com/public?command=returnTradeHistory&currencyPair=BTC_ETH&start=1484174782&end=1410158341'
request = requests.get(test)
data = request.json()
print(data)
"""
"""
with open('/home/mike/Documents/coding_all/machine_predict/pair_trading_history_data.csv', 'w') as f:
	writer = csv.DictWriter(f, pair_trades_key_list1)
	writer.writeheader()
	for coin in coin_list_test:
		# command term
		command_trade_history = command_trade_history = command_list[2] + coin\
	 + '&start=' + start_date + '&end=' + end_date
		# request using above command term
		return_history_response = requests.get('https://poloniex.com/public?command=' + command_trade_history)
		# dictionary return of coin pair put in 
		return_history_data = return_history_response.json()
		pair_dict = return_history_data
		#pair_dict2 = {'date_time_added_to_db':date_unix, 'coin_name':coin}
		#pair_dict.update(pair_dict2)
		writer.writerow(pair_dict)
		time.sleep(2)
"""

trade_dict = {}
for coin in coin_list_test:
	# command term
	command_trade_history = command_trade_history = command_list[2] + coin\
	 + '&start=' + start_date + '&end=' + end_date
	# request using above command term
	return_history_response = requests.get('https://poloniex.com/public?command=' + command_trade_history)
	# dictionary return of coin pair put in 
	return_history_data = return_history_response.json()
	pair_dict = return_history_data
	#print(pair_dict[0]['globalTradeID'])
	#for x in range(10):
		#print(return_history_data[x])
	# this is close to get indiv vars
	# then write row for each in db
	#trade_dict2 = {'coin_name':coin}
	print(type(pair_dict))
	for x in range(2):
		trade = pair_dict[0]
		for key in pair_trades_key_list1:
			#print(coin,x,key,pair_dict[x][key])
			value = pair_dict[x][key]
			trade_dict = {key:value}
			trade_dict2 = {'coin_name':coin}
			trade_dict.update(trade_dict2)
			print(trade_dict)

			#key = pair_dict[0]['globalTradeID']
			#print(trade, key)
			#writer.writerow(pair_dict)
