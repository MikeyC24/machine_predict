from ArrangeDataInOrder import *
import unittest

class TestArrangeData(unittest.TestCase):

	# file locations for test
	file_location_loan = '/home/mike/Documents/coding_all/data_sets_machine_predict/loans_2007.csv'
	#df = pd.read_csv(file_location)

	def test_fill_in_data_full_range(self):
		pass

	def test_group_by_time_with_vars(self):
		pass

	def test_drop_certain_percent_of_missing_data(self):
		df = pd.read_csv(self.file_location_loan)
		pre_len = len(df.columns.values)
		instance = ArrangeDataInOrder(df)
		instance.drop_certain_percent_of_missing_data(.99)
		post_len = len(instance.dataframe.columns.values)
		self.assertNotEqual(post_len, pre_len)

	def test_drop_columns_array(self):
		df = pd.read_csv(self.file_location_loan)
		pre_len = len(df.columns.values)
		instance = ArrangeDataInOrder(df)
		cols_to_drop = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'grade', 
		'sub_grade', 'emp_title', 'issue_d']
		instance.drop_columns_array(cols_to_drop)
		post_len = len((instance.dataframe.columns.values))+len(cols_to_drop)
		self.assertEqual(pre_len, post_len)

	def test_map_target_for_binary(self):
		df = pd.read_csv(self.file_location_loan)
		target = 'loan_status'
		yes = 'Fully Paid'
		no = 'Charged Off'
		instance = ArrangeDataInOrder(df)
		instance.map_target_for_binary(target, yes, no)
		does_not_equal = ['Current', 'In Grace Period',
		'Late (31-120 days)', 'Late (16-30 days)', 'Default',
		'Does not meet the credit policy. Status:Fully Paid',
		'Does not meet the credit policy. Status:Charged Off']
		for x in list(instance.dataframe['loan_status'].unique()):
			self.assertNotIn(x, does_not_equal)

	def test_drop_cols_with_one_unique_value(self):
		df = pd.read_csv(self.file_location_loan)
		pre = len(df.columns)
		instance = ArrangeDataInOrder(df)
		instance.drop_cols_with_one_unique_value()
		post = len(instance.dataframe.columns)
		self.assertEqual(pre, (post+5))

	def test_remove_col_by_percent_missing(self):
		df = pd.read_csv(self.file_location_loan)
		pre = len(df.columns)
		instance = ArrangeDataInOrder(df)
		instance.remove_col_by_percent_missing(.01)
		post = len(instance.dataframe.columns)
		self.assertEqual(pre, (post+2))

	def test_convert_to_numerical_for_percent(self):
		df = pd.read_csv(self.file_location_loan)
		instance = ArrangeDataInOrder(df)
		percent_num_convert = ['int_rate', 'revol_util']
		instance.convert_to_numerical(percent_num_convert, 'percent')
		self.assertTrue(instance.dataframe['int_rate'].dtype == 'float')
		self.assertTrue(instance.dataframe['revol_util'].dtype == 'float')

	def test_use_dict_map_for_new_cols(self):
		df = pd.read_csv(self.file_location_loan)
		instance = ArrangeDataInOrder(df)
		mapping_dict =  {
						    "emp_length": {
						        "10+ years": 10,
						        "9 years": 9,
						        "8 years": 8,
						        "7 years": 7,
						        "6 years": 6,
						        "5 years": 5,
						        "4 years": 4,
						        "3 years": 3,
						        "2 years": 2,
						        "1 year": 1,
						        "< 1 year": 0,
						        "n/a": 0
						    }
						}
		instance.use_dict_map_for_new_cols(mapping_dict)
		self.assertTrue(instance.dataframe['emp_length'].dtype == 'float')

	def test_convert_cols_to_dummies(self):
		cols_to_make_dummies = ['home_ownership', 'verification_status', 'purpose', 'term']
		df = pd.read_csv(self.file_location_loan)
		instance = ArrangeDataInOrder(df)
		pre = len(df.columns.values)
		instance.convert_cols_to_dummies(cols_to_make_dummies)
		post = len(instance.dataframe.columns.values)
		self.assertEqual(pre, (post-20))

	def test_drop_nan_values(self):
		cols_to_make_dummies = ['home_ownership', 'verification_status', 'purpose', 'term']
		df = pd.read_csv(self.file_location_loan)
		instance = ArrangeDataInOrder(df)
		pre = df.shape
		instance.drop_nan_values()
		post = instance.dataframe.shape
		print(pre, post)

if __name__ == '__main__':
	unittest.main()

