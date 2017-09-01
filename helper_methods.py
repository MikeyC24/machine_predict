import numpy as np

# check for groups of percent changes for var in data
def group_percent_changes(dataframe, var, window=1, forcast=1, up_value=.1, down_value=-.1):
	df = dataframe
	change = df[var].pct_change(window)
	change = change.replace([np.inf, -np.inf], np.nan)
	change = change.fillna(0.).values.tolist()
	#change = [c[0] for c in change]
	print(change)
	up_array = []
	down_array = []
	up_var_array = []
	down_var_array = []
	middle_var_array = []
	for x in change:
		up_array.append(x) if x > 0 else down_array.append(x)
	for x in change:
		if x > up_value:
			up_var_array.append(x)
		elif x < down_value:
			down_var_array.append(x)
		else:
			middle_var_array.append(x)
	dict_numbers = {}
	dict_numbers['up_values'] = up_array
	dict_numbers['down_values'] = down_array
	dict_numbers['up' +str(up_value) + '_value'] = up_var_array
	dict_numbers['down' +str(down_value) + '_value'] = down_var_array
	dict_numbers['middle' +str(up_value) + str(down_value) + '_value'] = middle_var_array
	return dict_numbers			
