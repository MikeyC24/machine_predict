#from MachinePredictModelrefractored import *
#from DatabaseFunctionality import *

#file_location = '/home/mike/Documents/coding_all/data_sets_machine_predict/cleaned_loans_2007.csv'
#df = pd.read_csv(file_location)
"""
drop_nan_rows = 'yes'
columns_to_drop = None
columns_all = ['loan_amnt' 'int_rate' 'installment' 'emp_length' 'annual_inc'
 'loan_status' 'dti' 'delinq_2yrs' 'inq_last_6mths' 'open_acc' 'pub_rec'
 'revol_bal' 'revol_util' 'total_acc' 'home_ownership_MORTGAGE'
 'home_ownership_NONE' 'home_ownership_OTHER' 'home_ownership_OWN'
 'home_ownership_RENT' 'verification_status_Not Verified'
 'verification_status_Source Verified' 'verification_status_Verified'
 'purpose_car' 'purpose_credit_card' 'purpose_debt_consolidation'
 'purpose_educational' 'purpose_home_improvement' 'purpose_house'
 'purpose_major_purchase' 'purpose_medical' 'purpose_moving'
 'purpose_other' 'purpose_renewable_energy' 'purpose_small_business'
 'purpose_vacation' 'purpose_wedding' 'term_ 36 months' 'term_ 60 months']
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
target = 'loan_status'
array_for_format_non_unix_date = None
#format_human_date = ['date', '%Y-%m-%d %H:%M:%S', 'UTC'] 
format_human_date = None
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
# need to add a shift for the rolling averages to accomadte window size
rolling_averages_dict = None
#rolling_averages_dict = { 'rate_USDT_ETH':[720],'rate_USDT_BTC':[24,360,720],'rate_USDT_LTC':[720]}
#rolling_averages_dict = { 'rate_USDT_BTC':[ 24, 360, 720]}
rolling_std_dict = None
#rolling_std_dict = {'rate_USDT_ETH':[24,48],'rate_USDT_BTC':[24],'rate_USDT_LTC':[24]}
#rolling_std_dict = {'rate_USDT_BTC':[24,720 ]}
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
					rolling_std_dict=rolling_std_dict)

print(df.head(10))
print(df.columns.values)
results = sample_instance.user_full_model()
for k,v in results.items():
	for kk, vv in v.items():
		print(k)
		print(kk)
		print(vv)
		print('_______________')
"""
from ArrangeDataInOrder import *
import sqlite3

file_location = '/home/mike/Documents/coding_all/data_sets_machine_predict/loans_2007.csv'
df = pd.read_csv(file_location)
print(df.head(2))
print(df.shape)
print(df.columns.values)
print(df.isnull().sum())

data_instace = ArrangeDataInOrder(df)
# redoing, lets get methods to return classs itself to 
# later avoid having to desginate what df is being used
print(data_instace.dataframe.shape)
print(data_instace.dataframe.columns.values)
# drop all columns missing a certain amount of data
df = data_instace.drop_certain_percent_of_missing_data(.5)
print(data_instace.dataframe.shape)
print(data_instace.dataframe.columns.values)
print(type(df.dataframe))
print(df.dataframe.shape)




"""
# drop unwanted columns 
cols_to_drop = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'grade', 'sub_grade', 'emp_title', 'issue_d',
'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
'total_rec_int', 'total_rec_late_fee', 'recoveries', 'last_pymnt_d', 'last_pymnt_amnt', 'collection_recovery_fee']
df = data_instace.drop_columns_array(cols_to_drop)
# pick target and see all values
target = 'loan_status'
print(df[target].value_counts())
# choose the 1 case, the 0 case and drop rest
yes = 'Fully Paid'
no = 'Charged Off'
df = data_instace.map_target_for_binary(target, yes, no)
# drop columns without enough unique values
df =  data_instace.drop_cols_with_one_unique_value()
# next task is to remove missing data, convert all to numerical, remove
# extra columns
print('show unique from class')
data_instace.show_unique_count_each_col()
print('null count no in class', df.isnull().sum())
print('show null count from class')
data_instace.show_nan_count(.01)
df = data_instace.remove_col_by_percent_missing(.01)
print('after remove')
data_instace.show_nan_count(dataframe_new=df)
df = data_instace.drop_nan_values()
#print(df.isnull().sum())
print('after dropping all nan')
data_instace.show_nan_count(dataframe_new=df)
data_instace.show_all_dtypes(dataframe_new=df)
"""

"""
# drop rows missing values or are nan
df = data_instace.drop_nan_values()
# drop remain rows missing data
df = data_instace.drop_nan_values()
print(df.head(10))
print(df.shape)
print(df.columns.values)
"""