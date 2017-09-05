from ArrangeDataInOrder import *
import sqlite3

# this page is to walk a user thru their data, ask them certain questions
# all to help prep the data and select the best choice
#skipping file read for now
file_location = '/home/mike/Documents/coding_all/data_sets_machine_predict/loans_2007.csv'
#file_location_coin = '/home/mike/Downloads/coin_months_data'
#con = sqlite3.connect(file_location_coin)
#table = 'top_3_jan_mid_aug_final'
#df = pd.read_sql_query('SELECT * FROM %s' % (table), con)
df = pd.read_csv(file_location)
data_instace = ArrangeDataInOrder(df)


# maybe set this up like the to d lust where it runs a loop and gives you option
# to keep manioulating as you go enter ready
# options could be see data, add columns (moving avg), remove (columns, missing rows)
# graphs/charts/story, calculate (avg, std) then rdy for models 
user_interface_input_options_show = ['add/change', 'remove', 'show', 'start_model']
user_interface_input_options = {'add/change':0, 'remove':1, 'show':2, 'start_model':3}
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
print('Now that your data is loaded and we have determined if it is a time series or not \
 . we now need to decide if that data needs to be arranged. type "add/change" or 1 to \
	manipulate rows/columns, type remove or 2 to drop columns/rows, type show or 3\
	to view certain parts of your data, and lastly type start model or 4 to run models')
user_interface_input = input('but first what type of model would you like to run: ')
# maybe add an explanation here or linear and classification
while user_interface_input != 'start model':
	prediction_type = input('Will your predication be classification or linear?: ')
	if prediction_type == 'linear':
		print('this model is not yet supported')
	elif prediction_type == 'classification':
		print('Lets get to work on the data, would you like to add/change, remove\
			or show')
		user_interface_input = input('please select: ')
		



	else:
		print('answer/model not recongized')
else:
	print('now going to model section')


# 2017-01-01 13:50:00,2017-08-07 12:40:00,10Min,date,yes
# 1H,no
