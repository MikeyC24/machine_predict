from DatabaseFull import *

# '/home/mike/Documents/coding_all/data_sets_machine_predict/coin_months_data'
location_base = '/home/mike/Documents/coding_all/data_sets_machine_predict/'
# 8/7/17 100 GMT
start_date =  '1502067600'
# 8/7/17 1300 GMT
end_date = '1502110800'
#database_name = 'pol_db_class_2gether_two'
# start date weekend
# July first
start_wke = '1496275200'
#end weekend
# August 11
end_wke = '1498867200'
time_interval_delta_measure = 'h'
time_interval_delta_amount = 3
top_3_coin_list = ['USDT_ETH', 'USDT_BTC', 'USDT_LTC']
# 2nd teir list USDT_XRP= ripple, Dash= USDT_DASH, Monero = USDT_XMR
list_two_coins = ['USDT_XRP', 'USDT_DASH', 'USDT_XMR']
coin_name_end = ''
## datbase clas vars
db_name = 'coin_months_data'
location_base1 = '/home/mike/Documents/coding_all/data_sets_machine_predict/'
table_name_array = ['USDT_BTC_table_', 'USDT_ETH_table_', 'USDT_LTC_table_']
columns_wanted_array = ['globalTradeID', 'date_time_added_to_db', 'coin_name', 'date', 'type', 'rate', 'amount', 'total']
columns_wanted_array1 = ['coin_name', 'date', 'rate', 'amount', 'total']
time_interval10 = '10Min'
write_to_db = 'yes'
write_to_db_tablename = '3_coin_list_june'
# here is where this db starts again
#start 2017-07-20 20:00:00 1500595200
#end 2017-07-20 21:00:00 1500598800
# error is from start 2017-07-20 09:00:00 1500555600 to same spot
# before new start date
# need to add some error exceptions
# underf for x in pais add a try and except, if error append error range to
# a array that is written to csv then go to next pair
# also need try except for pinging polinex, checking for bad connections
# bad urls, coins and or dates not in range, again append errors, write to csv
# and start next portion, for this for now, should also make it so
# all three coins are captured

#start_period_cycle, end_period_cycle, 
#time_period_interval, limit_interval_before_db_build,
#coin_list_array, db_name, coin_name_end, db_location_base, 
#database_name, table_name_array, cols_wanted_array, time_interval

# this is working for getting new data
data_class = DatabaseFull(location_base)
print('___________________________________')
result = data_class.cycle_over_dates_and_build_coin_db(start_wke, end_wke, 'H', 6,
					top_3_coin_list, db_name, coin_name_end,
					db_name, table_name_array, columns_wanted_array1, time_interval10,
					write_to_db, write_to_db_tablename)
for data in result:
	print('______________')
	print(data)
# info for appending databases
# table name 1 = BTC_LTC_ETH_July_2017
# 2 = BTC_LTC_ETH_July_21_2017_august_10_17

"""
combine_array = ['BTC_LTC_ETH_July_2017', 'BTC_LTC_ETH_July_21_2017_august_10_17']
combine_date = None
write_new_db_to_sql= 'yes'
new_sql_table_name = 'three_coin_multi_combine_nine'
combine_array_dict = [{'BTC_LTC_ETH_July_2017':None}, {'BTC_LTC_ETH_July_21_2017_august_10_17':None}, {'BTC_LTC_ETH_July_31_2017_august_10_17':None},{'BTC_LTC_ETH_August_2_17_august_10_17':None}]
new_combined_table = data_class.append_dataframes_dict(db_name, combine_array_dict,
		write_new_db_to_sql=write_new_db_to_sql,new_sql_table_name=new_sql_table_name)
print('______________________________')
print('made to end')
print(new_combined_table)
"""

"""
# this was for building mins and maxes
db_name_mm = 'test_min_max'
start_unix_min = '1502463600'
end_unix_min = '1502470800'
data_class.convert_trade_history_to_sql_start_end_vars(start_unix_min, end_unix_min, top_3_coin_list, db_name_mm, coin_name_end)
database_dict_min = data_class.aggregate_databases1(db_name_mm, table_name_array, columns_wanted_array1, time_interval10)
combined_dfs = data_class.merge_databases_for_models(db_name_mm, database_dict_min, write_to_db='yes', write_to_db_tablename='min_max_tables1')
for k,v in combined_dfs.items():
	print(k)
	print(v)
	print('______________')
"""

#database_dict = data_class.aggregate_databases1(db_name, table_name_array, columns_wanted_array1,
#	time_interval10)
"""
print('______________________________________')
print('end')
print('result[0]', type(result[0]), result[0])
print('______________________________________')
print('result[1]', type(result[1]), result[1])
print('______________________________________')
print('result[2]', type(result[2]), result[2])
print('______________________________________')
print('result[3]', type(result[3]), result[3])
print('_______________________________________')
"""
"""
for k,v in database_dict.items():
	print('k', k)
	print('v', v)
	print('_________________')
	"""

# playing around with error except on dates known not to work
# here is where this db starts again
#start 2017-07-20 20:00:00 1500595200
#end 2017-07-20 21:00:00 1500598800
# error is from start 2017-07-20 09:00:00 1500555600 to same spot
"""
location_base = '/home/mike/Documents/coding_all/data_sets_machine_predict/'
# 8/7/17 100 GMT
start_date =  '1502067600'
# 8/7/17 1300 GMT
end_date = '1500598800'
#database_name = 'pol_db_class_2gether_two'
# start date weekend
# July first
start_wke = '1500595200'
#end weekend
# August 1st
end_wke = '1500602400'
time_interval_delta_measure = 'h'
time_interval_delta_amount = 3
top_3_coin_list = ['USDT_ETH', 'USDT_BTC', 'USDT_LTC']
# 2nd teir list USDT_XRP= ripple, Dash= USDT_DASH, Monero = USDT_XMR
list_two_coins = 'USDT_XRP', 'USDT_DASH', 'USDT_XMR'
coin_name_end = ''
## datbase clas vars
db_name = 'coin_months_data_error_testing'
location_base1 = '/home/mike/Documents/coding_all/data_sets_machine_predict/'
table_name_array = ['USDT_BTC_table_', 'USDT_ETH_table_', 'USDT_LTC_table_']
columns_wanted_array = ['globalTradeID', 'date_time_added_to_db', 'coin_name', 'date', 'type', 'rate', 'amount', 'total']
columns_wanted_array1 = ['coin_name', 'date', 'rate', 'amount', 'total']
time_interval10 = '10Min'
write_to_db = 'yes'
write_to_db_tablename = 'BTC_LTC_ETH_error_ecept_test'
# here is where this db starts again
#start 2017-07-20 20:00:00 1500595200
#end 2017-07-20 21:00:00 1500598800
# error is from start 2017-07-20 09:00:00 1500555600 to same spot
# before new start date
# need to add some error exceptions
# underf for x in pais add a try and except, if error append error range to
# a array that is written to csv then go to next pair
# also need try except for pinging polinex, checking for bad connections
# bad urls, coins and or dates not in range, again append errors, write to csv
# and start next portion, for this for now, should also make it so
# all three coins are captured

#start_period_cycle, end_period_cycle, 
#time_period_interval, limit_interval_before_db_build,
#coin_list_array, db_name, coin_name_end, db_location_base, 
#database_name, table_name_array, cols_wanted_array, time_interval

# 2017-07-20 12:00:00

data_class = DatabaseFull(location_base)
print('___________________________________')
result = data_class.cycle_over_dates_and_build_coin_db(start_wke, end_wke, 'H', 1,
					top_3_coin_list, db_name, coin_name_end,
					db_name, table_name_array, columns_wanted_array1, time_interval10,
					write_to_db, write_to_db_tablename)
print(result[0])
"""