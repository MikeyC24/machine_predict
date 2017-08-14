from MachinePredictModelrefractored import *
from DatabaseFunctionality import *
from polniex_api_class import *
from pandas.api.types import is_numeric_dtype
"""
file_location = '/home/mike/Documents/coding_all/machine_predict/hour.csv'
df_bike = pd.read_csv(file_location)
columns_to_drop_bike = ['casual', 'registered', 'dteday']
columns_all_bike = ['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt_binary']
columns_all_bike_test = ['workingday','temp', 'cnt_binary', 'hr_new']
create_target_dict_bike = {'column_name_old':['cnt'], 'column_name_new':['cnt_binary'], 'value':[10]}
target_bike = 'cnt_binary'
set_multi_class_bike = ['hr', 6, 12, 18, 24 , 'hr_new']
random_state_bike = 1
training_percent_bike = .75
kfold_number_bike = 10 
cycle_vars_user_check = 'yes'
user_params_or_find_params= 'user'
minimum_feature_count_for_var_cycle = 2
logistic_regression_params_bike = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':'balanced', 'random_state':random_state_bike, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
# max depht and min samples leaf can clash 
decision_tree_params_bike = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':random_state_bike, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':'balanced', 'presort':False}
#decision_tree_params_loan = ['test']
nnl_params_bike = {'hidden_layer_sizes':(10, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':random_state_bike}
new_model_options = ['LogisticRegress', 'DecisionTreeCla', 'MLPClassifier(a']
kfold_dict = {'n_splits':10, 'random_state':random_state_bike, 'shuffle':False}
model_score_dict_all = {'LogisticRegress':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}, 'DecisionTreeCla':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}, 'MLPClassifier(a':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
model_score_dict_nnl = {'MLPClassifier(a':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
model_score_dict_log = {'LogisticRegress':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
model_score_dict_tree = {'DecisionTreeCla':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
model_score_dict_log_tree = {'DecisionTreeCla':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]},'LogisticRegress':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
user_optmize_input = ['class', 'constant', 'train', model_score_dict_log]
decision_tree_array_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':[None, 'balanced'], 'random_state':[random_state_bike]}
logistic_regression_array_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1], 'random_state':[random_state_bike]}
neural_net_array_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[True], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08], 'random_state':[random_state_bike]}
database_name = 'machine_predict_test_db'
table_name = 'bike_table5'
db_location_base = '/home/mike/Documents/coding_all/machine_predict/'
write_to_db = 'yes'
vars_to_return_from_db = ['\'roc_auc_score\'', '\'sensitivity\'', '\'fallout_rate\'']
#vars_to_return_from_db = ['roc_auc_score']
bike_predict = MachinePredictModel(df_bike, columns_all_bike_test, random_state_bike, training_percent_bike, kfold_number_bike, target_bike, cols_to_drop=columns_to_drop_bike,set_multi_class=set_multi_class_bike, target_change_bin_dict=create_target_dict_bike, kfold_dict=kfold_dict, param_dict_logistic=logistic_regression_params_bike, param_dict_decision_tree=decision_tree_params_bike, param_dict_neural_network=nnl_params_bike, param_dict_logistic_array=logistic_regression_array_vars, param_dict_decision_tree_array=decision_tree_array_vars, param_dict_neural_network_array=neural_net_array_vars, user_input_for_model_output=user_optmize_input, cycle_vars_user_check=cycle_vars_user_check, minimum_feature_count_for_var_cycle=minimum_feature_count_for_var_cycle, db_location_base=db_location_base, table_name=table_name, database_name= database_name, write_to_db = write_to_db, vars_to_return_from_db=vars_to_return_from_db)
data_wanted = bike_predict.user_full_model()
db_data = bike_predict.sort_database_results()
print('db_data', db_data)
"""



"""
print(type(data_wanted))
print(len(data_wanted))
for x,y in data_wanted.items():
	print(x)
	print('_______________________')
	for k,v in y.items():
		print(k)
		print(v)
		print('_____________________')
"""
"""
#combined databases
# vars
# full db for c/p '/home/mike/Documents/coding_all/data_sets_machine_predict/all_coin_history_db_big'
db_location_base = '/home/mike/Documents/coding_all/data_sets_machine_predict/'
db_name = 'all_coin_history_db_big'
table_name_array = ['USDT_BTC_table_', 'USDT_ETH_table_', 'USDT_LTC_table_']
columns_wanted_array = ['globalTradeID', 'date_time_added_to_db', 'coin_name', 'date', 'type', 'rate', 'amount', 'total']
columns_wanted_array1 = ['coin_name', 'date', 'rate', 'amount', 'total']
columns_wanted_array_test = ['coin_name', 'total']

database_instance = DatabaseFunctionality(db_location_base, db_name)
dbs = database_instance.aggregate_databases1(table_name_array, columns_wanted_array1)

combined_df = database_instance.merge_databases_for_models(dbs)
print(type(combined_df))
print(combined_df)
"""

"""
db_location_base = '/home/mike/Documents/coding_all/data_sets_machine_predict/'
db_name = '3_coin_test_db'
table_name_array = ['USDT_BTC_table_', 'USDT_ETH_table_', 'USDT_LTC_table_']
columns_wanted_array = ['globalTradeID', 'date_time_added_to_db', 'coin_name', 'date', 'type', 'rate', 'amount', 'total']
columns_wanted_array1 = ['coin_name', 'date', 'rate', 'amount', 'total']
columns_wanted_array_test = ['coin_name', 'total']
time_interval = '10Min'
write_to_db = 'yes'
write_to_db_tablename = 'aggregated_formatted_table'
database_instance = DatabaseFunctionality(db_location_base, db_name)
dbs = database_instance.aggregate_databases1(table_name_array, columns_wanted_array1, time_interval)
combined_df = database_instance.merge_databases_for_models(dbs,write_to_db=write_to_db,
											write_to_db_tablename=write_to_db_tablename)
print('dbs before combined')
for values in dbs.values():
	print(values.head(5))
"""

file_location = '/home/mike/Documents/coding_all/data_sets_machine_predict/coin_months_data'
#df = pd.read_csv(file_location)
con = sqlite3.connect(file_location)
table = 'second_coin_list_two'
df = pd.read_sql_query('SELECT * FROM %s' % (table), con)
drop_nan_rows = 'yes'
#columns_to_drop = None
columns_to_drop = ['amount_USDT_ETH','total_USDT_ETH', 'trade_count_USDT_ETH',
'min_rate_USDT_ETH', 'max_rate_USDT_ETH', 'rate_USDT_ETH', 'rate_USDT_ETH_change', 'date']
columns_to_drop = ['amount_USDT_ETH','total_USDT_ETH', 'trade_count_USDT_ETH',
'min_rate_USDT_ETH', 'max_rate_USDT_ETH', 'rate_USDT_ETH', 
'date', 'amount_USDT_BTC', 'total_USDT_BTC', 'amount_USDT_LTC', 'total_USDT_LTC']
# columns all before any editing 
columns_all_init = ['date']
# took date out of colums_all
columns_all = [ 'rate_USDT_BTC',  'amount_USDT_BTC',  'total_USDT_BTC', 
'trade_count_USDT_BTC', 'min_rate_USDT_BTC', 'max_rate_USDT_BTC', 'rate_USDT_ETH', 
'rate_USDT_LTC', 'amount_USDT_LTC', 'total_USDT_LTC', 'trade_count_USDT_LTC', 
'max_rate_USDT_LTC', 'max_rate_USDT_LTC', 'rate_USDT_ETH_binary']
#columns_all_test = ['workingday','temp', 'cnt_binary', 'hr_new']
#normalize_columns_array = ['rate_USDT_BTC',  'amount_USDT_BTC',  'total_USDT_BTC', 
#'trade_count_USDT_BTC', 'rate_USDT_LTC', 'amount_USDT_LTC',
#'total_USDT_LTC', 'trade_count_USDT_LTC',] 
normalize_columns_array = None
# these two became None because it was combined into one method and var
#time_period_returns_dict = {'column_name_old':['rate_USDT_ETH'], 'column_name_new':['rate_USDT_ETH_change'], 'freq':[72], 'shift':'yes'}
#create_target_dict = {'column_name_old':['rate_USDT_ETH_change'], 'column_name_new':['rate_USDT_ETH_change_binary'], 'value':[0]}
time_period_returns_dict = None
create_target_dict = None
#target = 'rate_USDT_ETH_change_binary'
create_target_in_one = {'target':['rate_USDT_ETH'], 'freq':[72], 'shift':'yes', 'value_mark':0, 'target_new':'rate_USDT_ETH_binary'}
target = create_target_in_one['target_new']
array_for_format_non_unix_date = ['date','%Y-%m-%d %H:%M:%S', 'UTC']
format_human_date = ['date', '%Y-%m-%d %H:%M:%S', 'UTC'] 
#format_human_date = None
convert_date_to_cats_for_class = ['date', 'US/Eastern', True]
convert_all_to_numeric = 'no'
columns_to_convert_to_dummy = ['days_of_week_US_Eastern', 'part_of_day_US_Eastern']
#columns_to_convert_to_dummy = None
#convert_date_to_cats_for_class = None
normalize_numerical_columns = 'no'
set_multi_class = None
random_state = 1
training_percent = .75
kfold_number = 3
cycle_vars_user_check = 'no'
minimum_feature_count_for_var_cycle = 4
logistic_regression_params = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':'balanced', 'random_state':random_state, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
decision_tree_params = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':random_state, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':'balanced', 'presort':False}
nnl_params = {'hidden_layer_sizes':(10, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':random_state}
kfold_dict = {'n_splits':10, 'random_state':random_state, 'shuffle':False}
model_score_dict_all = {'LogisticRegress':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.85,1], 'fallout_rate':[0,.3]}, 'DecisionTreeCla':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}, 'MLPClassifier(a':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
model_score_dict_nnl = {'MLPClassifier(a':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
model_score_dict_log = {'LogisticRegress':{'roc_auc_score':[.55,1], 'precision':[.4,1], 'significant_level':.05, 'sensitivity':[.8,1], 'fallout_rate':[0,.4]}}
model_score_dict_tree = {'DecisionTreeCla':{'roc_auc_score':[.055,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
model_score_dict_log_tree = {'DecisionTreeCla':{'roc_auc_score':[.55,1], 'precision':[.6,1], 'significant_level':.05, 'sensitivity':[.5,1], 'fallout_rate':[0,.4]},'LogisticRegress':{'roc_auc_score':[.55,1], 'precision':[.6,1], 'significant_level':.05, 'sensitivity':[.6,1], 'fallout_rate':[0,.3]}}
user_optmize_input = ['class', 'constant', 'train', model_score_dict_all]
decision_tree_array_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':[None, 'balanced'], 'random_state':[random_state]}
logistic_regression_array_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1], 'random_state':[random_state]}
neural_net_array_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[True], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08], 'random_state':[random_state]}
database_name = 'machine_predict_test_db'
table_name = 'coins_table1'
db_location_base = '/home/mike/Documents/coding_all/machine_predict/'
write_to_db = 'no'
#rolling_averages_dict = None
rolling_averages_dict = {'rate_USDT_LTC':[6,24,48,144], 'rate_USDT_BTC':[6,24,48,144]}
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
					rolling_averages_dict=rolling_averages_dict)

"""
# looking at data
result = sample_instance._set_up_data_for_prob_predict()
print(type(result))
print(result.dataframe.isnull().count())
print(result.dataframe.columns.values)
print(result.dataframe.dtypes)
print('_______________')
df =result.dataframe
print(df['rate_USDT_BTC'].shape[0])
#print(df['part_of_day_US_Eastern'].isnumeric())
print(is_numeric_dtype(df['rate_USDT_BTC']))
print(is_numeric_dtype(df['part_of_day_US_Eastern']))
print(df['part_of_day_US_Eastern'].dtype.kind)
print(df['part_of_day_US_Eastern'].unique())
print(df['days_of_week_US_Eastern'].unique())
print(df[df['rate_USDT_ETH_binary']==1].count())

"""
model = sample_instance.user_full_model()
for k,v in model.items():
	#print(k)
	for kk, vv in v.items():
		print(k)
		print(kk)
		print(vv)
		print('_______________________________')

"""
results = sample_instance.user_full_model()
print('made to end')
#print(results)

for k,v in results.items():
	#print(k)
	for kk, vv in v.items():
		print(k)
		print(kk)
		print(vv)
		print('_______________________________')
"""
# checking out data
"""
print(df.shape)
print(df.isnull().count())
df.dropna(inplace=True)
print(df.shape)
print(df.isnull().count())
"""
"""
needs to be done next
3. set the binary reults of yes or no to account for a future date
4. add trend lines, support, resistance levels, https://stackoverflow.com/questions/8587047/support-resistance-algorithm-technical-analysis
"""
"""
# this is for polinex and atabse class before refractoring 
# polniex vars
# '/home/mike/Documents/coding_all/data_sets_machine_predict/3_coin_test_db'
location_base = '/home/mike/Documents/coding_all/data_sets_machine_predict/'
# 8/7/17 100 GMT
start_date =  '1502067600'
# 8/7/17 1300 GMT
end_date = '1502110800'
#database_name = 'pol_db_class_2gether_two'
# start date weekend
# saturday 8/5/17 at 1200am GMT
start_wke = '1502236800'
#end weekend
# monday 8/7/17 1am GMT
end_wke = '1502251200'
time_interval_delta_measure = 'h'
time_interval_delta_amount = 3
top_3_coin_list = ['USDT_ETH', 'USDT_BTC', 'USDT_LTC']
coin_name_end = ''
## datbase clas vars
db_name = 'pol_data_combined_db_two'
location_base1 = '/home/mike/Documents/coding_all/data_sets_machine_predict/'
table_name_array = ['USDT_BTC_table_', 'USDT_ETH_table_', 'USDT_LTC_table_']
columns_wanted_array = ['globalTradeID', 'date_time_added_to_db', 'coin_name', 'date', 'type', 'rate', 'amount', 'total']
columns_wanted_array1 = ['coin_name', 'date', 'rate', 'amount', 'total']
columns_wanted_array_test = ['coin_name', 'total']
time_interval10 = '10Min'
write_to_db = 'yes'
write_to_db_tablename = 'poln_data_combined_final_table_three'

#start_period_cycle, end_period_cycle, 
#time_period_interval, limit_interval_before_db_build,
#coin_list_array, db_name, coin_name_end, db_location_base, 
#database_name, table_name_array, cols_wanted_array, time_interval



data_class = PolniexApiData(start_date,end_date,location_base)
result = data_class.cycle_over_dates_and_build_coin_db(start_wke, end_wke, 'H', 3,
					top_3_coin_list, db_name, coin_name_end, location_base1,
					db_name, table_name_array, columns_wanted_array1, time_interval10,
					write_to_db, write_to_db_tablename)

print('______________________________________')
print('end')
print('result[0]', type(result[0]), result[0])
print('______________________________________')
print('result[1]', type(result[1]), result[1])
print('______________________________________')
print('result[2]', type(result[2]), result[2])
print('______________________________________')
print('result[3]', type(result[3]), result[3])
"""