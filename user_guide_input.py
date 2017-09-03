from ArrangeDataInOrder import *
import sqlite3

# this page is to walk a user thru their data, ask them certain questions
# all to help prep the data and select the best choice
#skipping file read for now
file_location = None
file_location_coin = '/home/mike/Downloads/coin_months_data'
con = sqlite3.connect(file_location_coin)
table = 'top_3_jan_mid_aug_final'
df = pd.read_sql_query('SELECT * FROM %s' % (table), con)
data_instace = ArrangeDataInOrder(df)
#df = pd.read_csv(file_location)

time_series_input = []
#. 1. is the data a time series or not
time_series_question0 = ' Is this data a time series? This means, order and time matter \
and you will be predicting a next value some time out. type "yes" if that is the case\
otherwise enter "no": '
time_series_input.append(str(input(time_series_question0)))
print(time_series_input[0])
if time_series_input[0] == 'yes':
	time_series_question1 = 'Is this data missing any time gaps? If it is, enter a \
	start date, end date and time freq, index/date column in this format "start_date, \
	end_date, frequency, index, estimate". the correct date formate is 2017-08-07 12:40:00. \
	The correct time interval is 10Min or 1H. enter yes for estimate, if you \
	you would like missing values estimated. If is not missing data enter "no": '
	time_series_input.append(input(str(time_series_question1)).split(','))
	time_series_question2 = 'Would you like to group this data by a different time\
	interval? If no enter "no". If yes enter time interval and whether you would \
	like blank values to be estimated in yes or no fashion. enter format like\
	time_interval, estimate.: '
	time_series_input.append(input(str(time_series_question2)).split(','))
	for answer in time_series_input:
		print(type(answer))
		print(answer)
	try:
		if time_series_input[1][0] != 'no' or time_series_input[2][0] != 'no':
			print('preparing data')
			df = data_instace.fill_in_data_full_range(time_series_input[1][0], 
									time_series_input[1][1], time_series_input[1][2],
									index=time_series_input[1][3], interpolate=time_series_input[1][4])
			data_instace2  = ArrangeDataInOrder(df)
			df_new = data_instace2.group_by_time_with_vars(time_series_input[2][0], 
					reset_index='no', interpolate=time_series_input[2][1], 
					index='no', set_to_datetime='no')
		print(df_new.head(10))
	except:
		print('data does not neet time preparing or time preparing went wrong')
prediction_type = input('Will your predication be classification or linear?: ')
if prediction_type == 'linear':
	print('this model is not yet supported')
elif prediction_type == 'classification':
	print('please answer the follwoing qestions to best prepare the data')



else:
	print('answer/model not recongized')


# 2017-01-01 13:50:00,2017-08-07 12:40:00,10Min,date,yes
# 1H,no
