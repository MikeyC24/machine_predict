from ArrangeData import *
import unittest

# this testing is for the ArrangeData class however, it will only use test methods being used
# by MachinePredictModel at this point
# will at some point add a list of tested methods and ones not looked at
	
#prob should make test csv for this that has everything needed 
# csv file dataframe load
file_location = '/home/mike/Documents/coding_all/machine_predict/TestCSVForArrangeData.csv'
dataframe = pd.read_csv(file_location)
# ArrangeData class instance 
df = ArrangeData(dataframe)

""" methods to test used in MachinePredictModel
1. format_unix_date
# i think 2 does the same as one, skip for now
2. convert_unix_to_human_date(
# should prob refractor this one soon
3. resample_date
4. normalize_new_column
5. time_period_returns_dict
6. drop_columns
7. set_binary_from_dict
8. set_multi_class_array
"""

class TestArrangeData(unittest.TestCase):

	file_location = '/home/mike/Documents/coding_all/machine_predict/TestCSVForArrangeData.csv'
	dataframe = pd.read_csv(file_location)
	# ArrangeData class instance 
	df = ArrangeData(dataframe)

	def test_format_unix_date(self):
		date_column = 'date_unix'
		check = self.df.format_unix_date(date_column)
		test_check = (check['Datetime'].dtype == 'datetime64[ns]')
		test_check2 = (str(check['Datetime'][0]) == '2017-06-18 00:00:00')
		self.assertTrue(test_check)
		self.assertTrue(test_check2)

	# for now this method should output three new columns with the names. month_highs_avg
	# week_high_avgs and day_highs_avg
	def test_resample_data(self):
		column_old = 'USD_BTC_EX_High'
		df = self.df.resample_date(column_old, 'month_highs_avg', 'M', 'mean')
		df = self.df.resample_date(column_old, 'week_highs_avg', 'W', 'mean')
		df = self.df.resample_date(column_old, 'day_highs_avg', 'D', 'mean')
		self.assertEqual(round(df['month_highs_avg'][18]), 1899.0)
		self.assertEqual(round(df['week_highs_avg'][0]), 2645.0)
		self.assertEqual(round(df['day_highs_avg'][0]), 2570.0)

	# checks to make sure column has been noramlized by making sure all vaues are
	# between 0 and 1
	def test_normalize_new_column(self):
		col_norm = ['EUR_BTC_EX_High', 'USD_BTC_EX_High']
		new_col_names = ['EUR_BTC_EX_High_normalized', 'USD_BTC_EX_High_normalized' ]
		df = self.df.normalize_new_column(col_norm)
		for col in new_col_names:
			for x in range(df.shape[0]):
				self.assertGreaterEqual(df[col][x], 0)
				self.assertLessEqual(df[col][x], 1)

	# checks to if time period columns are returned in dataframe, needs to work
	# on columns from resample data and returns a frequency base on given var
	def test_time_period_returns_dict(self):
		column_old = 'USD_BTC_EX_High'
		time_period_returns_dict = {'column_name_old':['week_highs_avg', 'day_highs_avg'], 'column_name_new':['week_highs_avg_change', '3day_highs_avg_change'], 'freq':[1,3]}
		df = self.df.resample_date(column_old, 'week_highs_avg', 'W', 'mean')
		df = self.df.resample_date(column_old, 'day_highs_avg', 'D', 'mean')
		df = self.df.time_period_returns_dict(time_period_returns_dict)
		self.assertEqual(round(df['week_highs_avg_change'][7], 3), .053)
		self.assertEqual(round(df['3day_highs_avg_change'][3], 3), -.02)
		self.assertTrue(str(df['week_highs_avg_change'][2]) == 'nan')
		self.assertTrue(str(df['3day_highs_avg_change'][1]) == 'nan')

	# checks to make sure columns are actauly dropped from dataframe
	def test_drop_columns(self):
		self.df.overall_data_display(10)
		cols_to_drop1 = ['EUR_BTC_EX_High', 'USD_BTC_EX_High']
		print(cols_to_drop1)
		df2 = df.drop_columns(cols_to_drop1)
		print(df2)


#test_case = TestArrangeData()
#test_case.test_format_unix_date()
#test_case.test_resample_data()
#test_case.test_normalize_new_column()
#test_case.test_time_period_returns_dict()
#test_case.test_drop_columns()

# datetime64[ns]
#1899.4756129
#2645.25485714
#2570.312


file_location1 = '/home/mike/Documents/coding_all/machine_predict/TestCSVForArrangeData.csv'
file_location2 = '/home/mike/Documents/coding_all/machine_predict/hour.csv'
"""
dataframe1 = pd.read_csv(file_location1)
# ArrangeData class instance 
df = ArrangeData(dataframe1)
cols_to_drop1 = ['EUR_BTC_EX_High']
#test = df.drop_columns(cols_to_drop1)
test = df.shuffle_rows()
print(test.shape)
test = test.drop(['USD_BTC_EX_High'], axis = 1)
print(test.shape)
test = df.drop_columns(cols_to_drop1)
print(test.shape)
#test.overall_data_display(10)
#print(test.head(10))
print(type(test))
"""
drop = ['casual']
dataframe2 = pd.read_csv(file_location2)
df = ArrangeData(dataframe2)
check = df.shuffle_rows()
#print(check)
#df.overall_data_display(10)

"""
print(test.shape)
test = df.drop_columns(drop)
print(test.shape)
"""
def set_up_df():
	file_location2 = '/home/mike/Documents/coding_all/machine_predict/hour.csv'
	dataframe2 = pd.read_csv(file_location2)
	df = ArrangeData(dataframe2)
	df = df.shuffle_rows()
	return df

test = set_up_df()
test.overall_data_display(10)
"""
3 big things revealed
1. drop columns is not working on Arrange data, it is wiping out whole dataframe
2. first method in MachinePredictModelrefractored return an instance of
arrangedata class and not a df, dont know if this is desired yet
3. only some of the methods from arrange data are triggering from that first method
actaully all but drop columns from testing, it seems creating a now column will 
alter the db, need to see what happens if col stays the same

"""