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

""" 
other methods to test
check dummy variables on categories page in test data
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
		cols_to_drop1 = ['EUR_BTC_EX_High', 'USD_BTC_EX_High']
		df2 = self.df.drop_columns(cols_to_drop1)
		for col in cols_to_drop1:
			self.assertNotIn(col, df2.columns.values)

	# test to make sure target column is created and everything in that column
	# is 0 or 1 for binary classification
	def test_set_binary_from_dict(self):
		create_target_dict_bike = {'column_name_old':['cnt'], 'column_name_new':['cnt_binary'], 'value':[10]}
		target_bike = 'cnt_binary'
		df = self.df.set_binary_from_dict(create_target_dict_bike)
		self.assertIn(target_bike, df.columns.values)
		for x in range(df.shape[0]):
			y = True if (df[target_bike][x] == 0 or df[target_bike][x] == 1) else False
			self.assertTrue(y)

	# test to mak sure a column was turned into a multi class col, betweem the ranges
	# this may need to be changed as the method itself prob needs to be improved
	# this test is no tperfect and can miss if all values are same when they shpuldnt be
	def test_set_multi_class_array(self):
		set_multi_class_bike = ['hr', 6, 12, 18, 24 , 'hr_new']
		df = self.df.set_multi_class_array(set_multi_class_bike)
		self.assertIn(set_multi_class_bike[5], df.columns.values)
		for x in range(df.shape[0]):
			self.assertTrue(0 <= df[set_multi_class_bike[5]][x] <= 5)


test_case = TestArrangeData()
test_case.test_format_unix_date()
test_case.test_resample_data()
test_case.test_normalize_new_column()
test_case.test_time_period_returns_dict()
test_case.test_drop_columns()
test_case.test_set_binary_from_dict()
test_case.test_set_multi_class_array()


"""
3 big things revealed
1. drop columns is not working on Arrange data, it is wiping out whole dataframe - FIXED
2. first method in MachinePredictModelrefractored return an instance of
arrangedata class and not a df, dont know if this is desired yet
3. only some of the methods from arrange data are triggering from that first method
actaully all but drop columns from testing, it seems creating a now column will 
alter the db, need to see what happens if col stays the same

"""