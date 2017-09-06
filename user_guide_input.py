from ArrangeDataInOrder import *
import sqlite3

class UserMenu:

	#class vars/menus

	# method descriptions
	# add/change
	map_target_for_binary_info = 'This help creates a yes/no or in machine learning a 1,0 for \
	the target variable. To change, enter the target column name, the yes/1 variable in the column\
	and the no/1 variable in the order target,yes,no. This is case sensitive'
	#remove
	drop_certain_percent_missing_data_info = 'This method will drop all columns that is missing desired up to or greater\
	than the entered percent. Please provde a percent in decimal form '
	drop_columns_array_info = 'this method will drop the names of the columns entered\
	Please enter in the format col_1, col_2. This is case sensitive'
	#show
	show_data_frame_info = 'this will provide a snapshot of the dataframe, including the columns,\
	values in columns to desired number and the shape/dimensions. Please enter a number'
	show_unique_count_each_col_info = 'This method will show the column name and how many unique \
	values are in each column'

	master_menu_actions = {'add/change': {1:['map_target_for_binary', map_target_for_binary_info] },
						'remove':{1:['drop_certain_percent_missing_data', drop_certain_percent_missing_data_info],
								2:['drop_columns_array', drop_columns_array_info]},
						'show': {1:['show_data_frame', show_data_frame_info],
								2: ['show_unique_count_each_col', show_unique_count_each_col_info]},
						'enter_model':{1:['model, idk how this part will be done yet']}
						}


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
	time_series_question0 = 'Before we get started, we need to know.....\
	Is this data a time series? This means, order and time matter \
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
			if time_series_input[1][0] != 'no':
				print('preparing data')
				df_time = data_instace.fill_in_data_full_range(time_series_input[1][0], 
										time_series_input[1][1], time_series_input[1][2],
										index=time_series_input[1][3], interpolate=time_series_input[1][4])
			if time_series_input[2][0] != 'no':
				data_instace2  = ArrangeDataInOrder(df_time)
				df_time = data_instace2.group_by_time_with_vars(time_series_input[2][0], 
						reset_index='no', interpolate=time_series_input[2][1], 
						index='no', set_to_datetime='no')
			print(df.head(10))
		except:
			print('data does not neet time preparing or time preparing went wrong')
	if time_series_input[0] == 'yes':
		data_instace = ArrangeDataInOrder(df_time)
	else:
		data_instace = ArrangeDataInOrder(df)
	print('Now that your data is loaded and we have determined if it is a time series or not \
	 . we now need to decide if that data needs to be arranged.')
	prediction_type = input('but first what type of model would you like to run\
		 linear or classification: ').lower()
	# maybe add an explanation here or linear and classification
	print('you have chose ' + prediction_type + 'Before we jump into the model. Lets \
		see if we need to arrange the data. You will have the option to add/change the data,\
		 remove data, and be able to see certain characteristics of the data. At any point\
		 type in 4 to jump into the model, but dont worry, before you jump into the \
		 model you will have the option to save your newly manipulated data into a \
		 csv file to save right to sql. Lets get started')
	user_interface_input = 0
	while user_interface_input != 4:
		if prediction_type == 'linear':
			print('this model is not yet supported')
		elif prediction_type == 'classification':
			print('Lets get to work on the data, press 1 to bring up the options on add/\
				change the data, 2 to remove data, 3 to show the data, or 4 to start\
				the models. (you can save data before model')
			user_interface_input = int(input('please select: '))
			print(user_interface_input)
			print(type(user_interface_input))
			if user_interface_input == 1:
				print('made it under choice 1')
				# cant get to next part after one is chosen
				for x,k in master_menu_actions['add/change'].items():
					print('select the number for the method you would like and follow the prompt')
					print(x, 'the method is ' + k[0], 'description: ' + k[1] )
				print('under for loop choice 1')
			else:
				print('That is not a menu option')



		else:
			print('answer/model not recongized')
			break
	else:
		print('now going to model section')


	# 2017-01-01 13:50:00,2017-08-07 12:40:00,10Min,date,yes
	# 1H,no

	# different menus



if __name__ == '__main__':
	print('ran program')
