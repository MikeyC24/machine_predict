from DatabaseFull import *


location_base = '/home/mike/Documents/coding_all/data_sets_machine_predict/'
# 8/7/17 100 GMT
start_date =  '1502067600'
# 8/7/17 1300 GMT
end_date = '1502110800'
#database_name = 'pol_db_class_2gether_two'
# start date weekend
# saturday 8/5/17 at 1200am GMT
start_wke = '1502236800'
#end weekend
# monday 8/7/17 1am GMT
end_wke = '1502280000'
time_interval_delta_measure = 'h'
time_interval_delta_amount = 3
top_3_coin_list = ['USDT_ETH', 'USDT_BTC', 'USDT_LTC']
coin_name_end = ''
## datbase clas vars
db_name = 'pol_data_combined_db_two'
location_base1 = '/home/mike/Documents/coding_all/data_sets_machine_predict/'
table_name_array = ['USDT_BTC_table_', 'USDT_ETH_table_', 'USDT_LTC_table_']
columns_wanted_array = ['globalTradeID', 'date_time_added_to_db', 'coin_name', 'date', 'type', 'rate', 'amount', 'total']
columns_wanted_array1 = ['coin_name', 'date', 'rate', 'amount', 'total']
columns_wanted_array_test = ['coin_name', 'total']
time_interval10 = '10Min'
write_to_db = 'yes'
write_to_db_tablename = 'poln_data_combined_final_table_three'

#start_period_cycle, end_period_cycle, 
#time_period_interval, limit_interval_before_db_build,
#coin_list_array, db_name, coin_name_end, db_location_base, 
#database_name, table_name_array, cols_wanted_array, time_interval



data_class = DatabaseFull(location_base)
print('___________________________________')
result = data_class.cycle_over_dates_and_build_coin_db1(start_wke, end_wke, 'H', 6,
					top_3_coin_list, db_name, coin_name_end,
					db_name, table_name_array, columns_wanted_array1, time_interval10,
					write_to_db, write_to_db_tablename)
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