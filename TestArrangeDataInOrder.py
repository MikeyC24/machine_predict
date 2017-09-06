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
		post_len = len((instance.dataframe.columns.values))-8
		self.assertEqual(pre_len, post_len)

if __name__ == '__main__':
	unittest.main()

