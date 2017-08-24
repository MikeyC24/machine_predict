from MachinePredictModelrefractored import *
#from KerasClass import *
import unittest

# may be worth looking into for storing large datasets https://github.com/fchollet/keras/issues/68


file_location = '/home/mike/Documents/coding_all/data_sets_machine_predict/coin_months_data'
file_location1 = '/home/mike/Downloads/coin_months_data'
#df = pd.read_csv(file_location)
con = sqlite3.connect(file_location)
table1 = 'second_coin_list_two'
table = 'top_3_jan_mid_aug_final'
df = pd.read_sql_query('SELECT * FROM %s' % (table), con)
"""
# to change time interval
arrange_data_for_df_timeframe = ArrangeData(df)
df_by_hour = arrange_data_for_df_timeframe.group_by_time_with_vars('1H',reset_index='no')
df = df_by_hour
# I think this is the new way to format data
# maybe make this a class to prep data for machine predict and all machine predict classes have no index
data_instace  = ArrangeData(df)
filled_df = data_instace.fill_in_data_full_range(start_date, end_date, '10Min',
										index='date', interpolate='yes')
print(filled_df.head(10), filled_df.shape)

data_instace2  = ArrangeData(filled_df)
hourly_df = data_instace2.group_by_time_with_vars('1H', reset_index='no', index='no'
										, set_to_datetime='no')
"""
# selecting new df to accomade for all timeslots since jan and make hourly
start_date = '2017-01-01 13:50:00'
end_date = '2017-08-07 12:40:00'
data_instace  = ArrangeData(df)
filled_df = data_instace.fill_in_data_full_range(start_date, end_date, '10Min',
										index='date', interpolate='yes')
print(filled_df.head(10), filled_df.shape)

data_instace2  = ArrangeData(filled_df)
hourly_df = data_instace2.group_by_time_with_vars('4H', reset_index='no', index='no'
										, set_to_datetime='no')
print(hourly_df.head(10), hourly_df.shape)
print('___________')
print(hourly_df.index[0])
print(hourly_df.index[-1])
df= hourly_df