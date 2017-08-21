import sqlite3
import pandas as pd
import numpy as np


# method to split dataframe into multiple by column and recpombine later
conn = sqlite3.connect('/home/mike/Documents/coding_all/data_sets_machine_predict/db_array_rearrange_two')
df = pd.read_sql_query('SELECT * FROM %s' % ('x_test_table_1'), conn)
print(df.columns.values)
print(type(df))
df1 = df.loc[:,:'13']
df2= df.loc[:,'14':]
print(df1.head(10)) 
print(df2.head(10)) 
df3 = pd.concat([df1,df2], axis =1)
print(df3.head(10)) 

df4 = df.loc[:,:'14']
df5= df.loc[:,'15':'29']
df6= df.loc[:,'30':]
df_array = [df4, df5, df6]
df7 = pd.concat(df_array, axis =1)
print(df.head(10))
print(df3.head(10))
a= True if df.equals(df7) else False
print(a)

def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out

def separate_dfs_by_cols_even(df, num):
	cols = df.columns.values
	col_len = len(df.columns.values)
	print(cols, col_len)
	chunks = chunkIt(cols, num)
	print(chunks)
	counter = 1
	dfs_array = []
	for array in chunks:
		print(array)
		name = 'df' + str(counter)
		print(array[0])
		print(array[-1])
		name = df.loc[:,array[0]:array[-1]]
		dfs_array.append(name)
		counter +=1
	return dfs_array 

dfs = separate_dfs_by_cols_even(df, 6)
for x in dfs:
	print(x.columns.values)

df8 = pd.concat(dfs, axis =1)
a= True if df.equals(df8) else False
print(df.head(5))
print(df8.head(5))
print(a)