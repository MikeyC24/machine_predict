

def time_period_returns_array(dict_vars):
	#df = self.dataframe
	#for keys, values in dict_vars.items():
	#range_count = int(len(list(dict_vars.values()))/len(list(dict_vars.keys())))
	#print(len(list(dict_vars.values())))
	#print(range_count)
	#range_count = range_count+1
	#print(range_count)
	count = sum(len(v) for v in dict_varss1.values())
	range_count = int(count / len(list(dict_vars.keys())))
	print(count)
	for x in range(range_count):
		#print(keys, values)
		print(x)
		print(dict_vars['column_name_old'][x])
		print(dict_vars['column_name_new'][x])
		print(dict_vars['freq'][x])
		#prices = df[column_name_old]
		#print(type(prices))
		#df[column_name_new] = prices.pct_change(freq)
	#return df


dict_varss = {'column_name_old': 'colold', 'column_name_new':'colnew', 'freq':1, 'column_name_old': 'colold1', 'column_name_new':'colnew1', 'freq':3}
dict_varss1 = {'column_name_old':['colold', '2ndcol'], 'column_name_new':['colnew', '2ndcol'], 'freq':[1,2]}
time_period_returns_array(dict_varss1)
count = sum(len(v) for v in dict_varss1.values())
#print(count)

"""
print(len(dict_varss))
#count = sum(len(v) for v in dict_varss.values())
#print(count)
countv = list(dict_varss.keys())
print(countv)
print(len(countv))
print(len(countv)/3)
#time_period_returns_array(dict_varss)
"""