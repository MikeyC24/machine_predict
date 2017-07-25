from ArrangeData import *
import unittest

# this testing is for the ArrangeData class however, it will only use test methods being used
# by MachinePredictModel at this point
# will at some point add a list of tested methods and ones not looked at

#prob should make test csv for this that has everything needed 
# csv file dataframe load
file_location = '/home/mike/Documents/coding_all/machine_predict/hour.csv'
df_bike = pd.read_csv(file_location)
# ArrangeData class instance 
df_bike = ArrangeData(df_bike)

def test_format_unix_date():
	date_column = 'dteday'
	print(type(df))
	df.
	#print(df[date_column])

test_format_unix_date()