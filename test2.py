from MachinePredictModelrefractored import *
#from KerasClass import *
import unittest
from ArrangeData import *

# may be worth looking into for storing large datasets https://github.com/fchollet/keras/issues/68


file_location = '/home/mike/Documents/coding_all/data_sets_machine_predict/coin_months_data'
file_location1 = '/home/mike/Downloads/coin_months_data'
#df = pd.read_csv(file_location)
con = sqlite3.connect(file_location)
table1 = 'second_coin_list_two'
table = 'top_3_jan_mid_aug_final'
df = pd.read_sql('SELECT * FROM %s' % (table), con)
drop_nan_rows = 'yes'
#columns_to_drop = None
#columns_to_drop = ['amount_USDT_ETH','total_USDT_ETH', 'trade_count_USDT_ETH',
#'min_rate_USDT_ETH', 'max_rate_USDT_ETH', 'rate_USDT_ETH', 'rate_USDT_ETH_change', 'date']
columns_to_drop1 = ['amount_USDT_ETH','total_USDT_ETH',
 'amount_USDT_BTC', 'total_USDT_BTC', 'amount_USDT_LTC', 'total_USDT_LTC', 'date',
 'trade_count_USDT_LTC', 'max_rate_USDT_LTC','rate_USDT_BTC',
 'trade_count_USDT_BTC', 'min_rate_USDT_BTC', 'max_rate_USDT_BTC', 'rate_USDT_LTC',
 'min_rate_USDT_LTC', ]
columns_to_drop = ['amount_USDT_ETH','total_USDT_ETH',
 'amount_USDT_BTC', 'total_USDT_BTC', 'amount_USDT_LTC', 'total_USDT_LTC']
# columns all before any editing 
columns_all_init = ['date']
# took date out of colums_all
columns_all = [ 'rate_USDT_BTC',  'amount_USDT_BTC',  'total_USDT_BTC', 
'trade_count_USDT_BTC', 'min_rate_USDT_BTC', 'max_rate_USDT_BTC', 'rate_USDT_ETH', 
'rate_USDT_LTC', 'amount_USDT_LTC', 'total_USDT_LTC', 'trade_count_USDT_LTC', 
'max_rate_USDT_LTC', 'max_rate_USDT_LTC', 'date']
group_by_time_with_vars = '1H'
#columns_all_test = ['workingday','temp', 'cnt_binary', 'hr_new']
#normalize_columns_array = ['rate_USDT_BTC',  'amount_USDT_BTC',  'total_USDT_BTC', 
#'trade_count_USDT_BTC', 'rate_USDT_LTC', 'amount_USDT_LTC',
#'total_USDT_LTC', 'trade_count_USDT_LTC',] 
normalize_columns_array = None
# these two became None because it was combined into one method and var
#time_period_returns_dict = {'column_name_old':['rate_USDT_ETH'], 'column_name_new':['rate_USDT_ETH_change'], 'freq':[1], 'shift':'no'}
#create_target_dict = {'column_name_old':['rate_USDT_ETH_change'], 'column_name_new':['rate_USDT_ETH_change_binary'], 'value':[0]}
time_period_returns_dict = None
create_target_dict = None
#target = 'rate_USDT_ETH_change_binary'
create_target_in_one = None
target = 'rate_USDT_ETH'
array_for_format_non_unix_date = ['date','%Y-%m-%d %H:%M:%S', 'UTC']
format_human_date = ['date', '%Y-%m-%d %H:%M:%S', 'UTC'] 
#format_human_date = None
convert_date_to_cats_for_class = None
convert_all_to_numeric = 'no'
columns_to_convert_to_dummy = None
#columns_to_convert_to_dummy = None
#convert_date_to_cats_for_class = None
normalize_numerical_columns = 'no'
#cat_rows_for_time_delta = ['date', 6, True]
cat_rows_for_time_delta = None
set_multi_class = None
random_state = 1
training_percent = .75
kfold_number = 10
cycle_vars_user_check = 'no'
minimum_feature_count_for_var_cycle = 4
logistic_regression_params = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':'balanced', 'random_state':random_state, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
decision_tree_params = {'criterion':'entropy', 'splitter':'best', 'max_depth':10, 'min_samples_split':8, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':'auto', 'random_state':random_state, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':'balanced', 'presort':False}
nnl_params = {'hidden_layer_sizes':(10, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':random_state}
kfold_dict = {'n_splits':10, 'random_state':random_state, 'shuffle':False, 'stratified':'yes'}
model_score_dict_all = {'LogisticRegress':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.85,1], 'fallout_rate':[0,.3]}, 'DecisionTreeCla':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}, 'MLPClassifier(a':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
model_score_dict_nnl = {'MLPClassifier(a':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
model_score_dict_log = {'LogisticRegress':{'roc_auc_score':[.55,1], 'precision':[.4,1], 'significant_level':.05, 'sensitivity':[.8,1], 'fallout_rate':[0,.4]}}
model_score_dict_tree = {'DecisionTreeCla':{'roc_auc_score':[.055,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
model_score_dict_log_tree = {'DecisionTreeCla':{'roc_auc_score':[.55,1], 'precision':[.6,1], 'significant_level':.05, 'sensitivity':[.5,1], 'fallout_rate':[0,.4]},'LogisticRegress':{'roc_auc_score':[.55,1], 'precision':[.6,1], 'significant_level':.05, 'sensitivity':[.6,1], 'fallout_rate':[0,.3]}}
user_optmize_input = ['class', 'constant', 'simple', model_score_dict_all]
decision_tree_array_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':['balanced'], 'random_state':[random_state]}
logistic_regression_array_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1], 'random_state':[random_state]}
neural_net_array_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[True], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08], 'random_state':[random_state]}
database_name = 'machine_predict_test_db'
table_name = 'coins_table1'
db_location_base = '/home/mike/Documents/coding_all/machine_predict/'
write_to_db = 'no'
#rolling_averages_dict = None
rolling_averages_dict = { 'rate_USDT_ETH':[10,30],'rate_USDT_ETH':[10,30],'rate_USDT_ETH':[10,30]}
rolling_std_dict = {'rate_USDT_ETH':[10,30],'rate_USDT_ETH':[10,30],'rate_USDT_ETH':[10,30]}
# sample instance has all vars above in it 
sample_instance = MachinePredictModel(df, columns_all, random_state, 
					training_percent, kfold_number, target, drop_nan_rows=drop_nan_rows,
					cols_to_drop=columns_to_drop, set_multi_class=set_multi_class, 
					target_change_bin_dict=create_target_dict, kfold_dict=kfold_dict,
					format_human_date = format_human_date,
					convert_date_to_cats_for_class=convert_date_to_cats_for_class,
					convert_all_to_numeric=convert_all_to_numeric,
					columns_to_convert_to_dummy=columns_to_convert_to_dummy,
					time_period_returns_dict=time_period_returns_dict,
					normalize_numerical_columns=normalize_numerical_columns,
					create_target_in_one=create_target_in_one,
					cat_rows_for_time_delta=cat_rows_for_time_delta,
					param_dict_logistic=logistic_regression_params, 
					param_dict_decision_tree=decision_tree_params, 
					param_dict_neural_network=nnl_params, 
					param_dict_logistic_array=logistic_regression_array_vars, 
					param_dict_decision_tree_array=decision_tree_array_vars, 
					param_dict_neural_network_array=neural_net_array_vars, 
					user_input_for_model_output=user_optmize_input, 
					cycle_vars_user_check=cycle_vars_user_check, 
					minimum_feature_count_for_var_cycle=minimum_feature_count_for_var_cycle,
					database_name=database_name, table_name=table_name, db_location_base=db_location_base,
					write_to_db=write_to_db, normalize_columns_array=normalize_columns_array,
					rolling_averages_dict=rolling_averages_dict,
					rolling_std_dict=rolling_std_dict,
					group_by_time_with_vars=group_by_time_with_vars)
"""
result = sample_instance._set_up_data_for_prob_predict()
df =result.dataframe
print('_______________')
#df = df.set_index('date')
print(df.head(10))
print(df.shape)
print(df.iloc[0])
print(df.iloc[-1])
#print(df.columns.values)
"""
"""
print('___________')
df = df.reindex()
print(df.head(10))
print(df.shape)
"""
"""
start_date = '2017-01-01 13:50:00'
end_date = '2017-08-07 12:40:00'

drange = pd.date_range(start=start_date, end=end_date, freq='10Min')
print(drange)
#idx = pd.date_range('09-01-2013 00:00:00', '09-30-2013 00:00:00', freq='12H')
#print(idx)
#df.reset_index(drop=True, inplace=True)
#print(df.head(10))
cols = ['max_rate_USDT_BTC' 'max_rate_USDT_ETH' 'max_rate_USDT_LTC'
 'min_rate_USDT_BTC' 'min_rate_USDT_ETH' 'min_rate_USDT_LTC'
 'rate_USDT_BTC' 'rate_USDT_ETH' 'rate_USDT_LTC' 'trade_count_USDT_BTC'
 'trade_count_USDT_ETH' 'trade_count_USDT_LTC' 'MA_10_rate_USDT_ETH'
 'MA_30_rate_USDT_ETH' 'MA_STD10_rate_USDT_ETH' 'MA_STD30_rate_USDT_ETH']
 # https://stackoverflow.com/questions/25909984/missing-data-insert-rows-in-pandas-and-fill-with-nan
#df = df.drop('date', axis=1)
#print(df.columns.values)
dseries = df['date'].values
df = df.drop('date', axis=1)
df.reset_index(drop=True, inplace=True)
print(df.index.is_unique)
print(df.head(10))
df['date_col'] = dseries
print(df.index.is_unique)
print(df.head(10))
#df.set_index('date_col',inplace=True, drop=True)
#print(df.index.is_unique)
df.reset_index(drop=True, inplace=True)
print('________________one')
print(df.head(10))
print(df.index.is_unique)
print(df.shape)
len_drange = len(drange)
print('____________ two')
df = df.reindex(range(len_drange))
print(df.head(10))
print(df.isnull().sum())
df['new_date'] = drange
print('_________________three')
print(df.head(10))
print(df.shape)
df.set_index('new_date', inplace=True)
print(df.head(10))
print(df.shape)
print(df.isnull().sum())
print(df)

#print(len_drange)
#print(df.head(10))
"""
"""
print('____________________')
print(df.head(10))
print(df.index.is_unique)
print('___________________')
df.reindex(drange)
print(df.head(10))
#print(dseries)
"""
"""
df = df.set_index('date', drop=True)
print(df.head(10))
print(df.columns.values)
print(df.index.is_unique)
print(df.index)
df.reset_index()
print(df.index.is_unique)
"""
"""
df.reset_index(inplace=True)
print(df.index.is_unique)
print(df.head(10))
df.set_index('date', inplace=True, drop=True)
print(df.head(10))
print(df.index.is_unique)
print(df.index.unique)
"""
#df.index = pd.DatetimeIndex(df.index)
#print(df.head(10), df.shape)
#df = df.reindex(drange)
#print(df.head(10))

#print(df.head(10), df.shape)

cols = ['date', 'amount_USDT_BTC', 'amount_USDT_ETH', 'amount_USDT_LTC',
 'max_rate_USDT_BTC', 'max_rate_USDT_ETH', 'max_rate_USDT_LTC',
 'min_rate_USDT_BTC', 'min_rate_USDT_ETH', 'min_rate_USDT_LTC',
 'rate_USDT_BTC', 'rate_USDT_ETH', 'rate_USDT_LTC', 'total_USDT_BTC',
 'total_USDT_ETH', 'total_USDT_LTC', 'trade_count_USDT_BTC',
 'trade_count_USDT_ETH', 'trade_count_USDT_LTC']

print(df.head(10))
start_date = '2017-01-01 00:00:00'
end_date = '2017-08-07 12:40:00'
"""
drange = pd.date_range(start=start_date, end=end_date, freq='10Min')
len_drange = len(drange)
print(drange)
#df.to_csv('test_csv')
#df1 = pd.read_csv('test_csv', index_col='date')
#print(df.head(10))
print(df.shape)
print(df['date'].dtype)
df1= df.set_index('date')
df1.index = pd.DatetimeIndex(df1.index)
print(df1.head(10))
print(df1.index.dtype)
print(df1[df1.index.duplicated()])
print('__________')
df1 = df1.groupby(df1.index).first()
print(df1.head(10))
print(df1.shape)
print(df1[df1.index.duplicated()])
df1 = df1.reindex(drange)
print(df1.head(10))
print(df1.shape)
print(df1.isnull().sum())
df1 = df1.interpolate()
print(df1.head(10))
print(df1.shape)
print(df1.isnull().sum())
"""
#df.set_index('date', inplace=True)
#print(df.head(10))
#print('____________')
#print(df.index[0])
#print(df.index[-1])



def fill_in_data_full_range(dataframe, index, start_date, end_date, freq, interpolate):
	if index != 'no':
		df = dataframe.set_index(index)
	else:
		df = dataframe
	#print(df.head(10))
	# make datetime
	df.index = pd.DatetimeIndex(df.index)
	# get rid of dups
	df = df.groupby(df.index).first()
	# set new range 
	drange = pd.date_range(start=start_date, end=end_date, freq=freq)
	#print(len(drange))
	df = df.reindex(drange)
	if interpolate == 'yes':
		df = df.interpolate()
	return df
new_df = fill_in_data_full_range(df, 'date', start_date, end_date, '10Min', 'no' )
print(new_df.head(10), new_df.shape)



data_instace  = ArrangeData(df)
filled_df = data_instace.fill_in_data_full_range(start_date, end_date, '10Min',
										index='date', interpolate='yes')
print(filled_df.head(10), filled_df.shape)

data_instace2  = ArrangeData(filled_df)
hourly_df = data_instace2.group_by_time_with_vars('1H', reset_index='no', index='no'
										, set_to_datetime='no')
#print(hourly_df.head(10))
print(hourly_df.head(-10))
print(hourly_df.shape)

start_date = '2017-01-01 13:50:00'
end_date = '2017-08-07 12:40:00'

drange = pd.date_range(start=start_date, end=end_date, freq='10Min')
#print(len(drange))

# formula to fill out index...... set index to date, make it unqiue, remove duplicates,
# crafting way to do this = df1 = df1.groupby(df1.index).first()
# then reindex on new date range, then interpolate

"""
df1.drop_duplicates(['min_rate_USDT_BTC'], inplace=True, keep='last')
#df1 = df1.drop(df1.loc['2017-07-17 01:00:00'])
print(df1[df1.index.duplicated()])
df1.drop(df1.loc['2017-07-17 01:00:00'], axis=0)
print(df1.loc['2017-07-20 12:00:00'])
print('_________')
"""


#df1 = df1.reindex(drange, fill_value='NaN')
"""
df1.index = pd.DatetimeIndex(df1.index)
df1 = df1.reindex(drange, fill_value='NaN')
df1.head(10)
"""