from MachinePredictModelrefractored import *

# going foward everything will run off of this page. there is a variable template at start
# it is between """ """ so it doesnt run but it has every possible variable combo
"""
all vars 
#file_location = '/home/mike/Documents/coding_all/machine_predict/hour.csv'
#df = pd.read_csv(file_location)
df = combined_df
columns_to_drop = ['amount_USDT_ETH','total_USDT_ETH', 'trade_count_USDT_ETH']
columns_all = ['date', 'rate_USDT_BTC',  'amount_USDT_BTC',  'total_USDT_BTC', 
'trade_count_USDT_BTC',  'rate_USDT_ETH', 'rate_USDT_LTC', 'amount_USDT_LTC','total_USDT_LTC', 'trade_count_USDT_LTC']
#columns_all_test = ['workingday','temp', 'cnt_binary', 'hr_new']
time_period_returns_dict = {'column_name_old':'rate_USDT_ETH', 'column_name_new':'rate_USDT_ETH_change', 'freq':1}
create_target_dict = {'column_name_old':['rate_USDT_ETH_change'], 'column_name_new':['rate_USDT_ETH_change_binary'], 'value':[.01]}
target = 'rate_USDT_ETH_change_binary'
#set_multi_class = ['hr', 6, 12, 18, 24 , 'hr_new']
random_state = 1
training_percent = .08
kfold_number = 10 
cycle_vars_user_check = 'no'
minimum_feature_count_for_var_cycle = 3
logistic_regression_params = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':'balanced', 'random_state':random_state, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
decision_tree_params = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':random_state, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':'balanced', 'presort':False}
nnl_params = {'hidden_layer_sizes':(10, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':random_state}
kfold_dict = {'n_splits':10, 'random_state':random_state, 'shuffle':False}
model_score_dict_all = {'logistic':{'roc_auc_score':.03, 'precision':.06, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}, 'decision_tree':{'error_metric':'roc_auc_score', 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}, 'neural_network':{'error_metric':'roc_auc_score', 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
model_score_dict_nnl = {'neural_network':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
model_score_dict_log = {'logistic':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
model_score_dict_tree = {'decision_tree':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
model_score_dict_log_tree = {'decision_tree':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]},'logistic':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
user_optmize_input = ['class', 'optimize', 'train', model_score_dict_log]
decision_tree_array_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':[None, 'balanced'], 'random_state':[random_state]}
logistic_regression_array_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1], 'random_state':[random_state]}
neural_net_array_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[True], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08], 'random_state':[random_state]}
database_name = 'machine_predict_test_db'
table_name = 'bike_table1'
db_location_base = '/home/mike/Documents/coding_all/machine_predict/'
write_to_db = 'no'
# sample instance has all vars above in it
sample_instance = MachinePredictModel(df, columns_all, random_state, 
					training_percent, kfold_number, target, 
					cols_to_drop=columns_to_drop, set_multi_class=set_multi_class, 
					target_change_bin_dict=create_target_dict, kfold_dict=kfold_dict,
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
					 write_to_db=write_to_db)
results = sample_instance.user_full_model()
"""
"""
# info for bikes
file_location = '/home/mike/Documents/coding_all/machine_predict/hour.csv'
df_bike = pd.read_csv(file_location)
columns_to_drop_bike = ['casual', 'registered', 'dteday']
columns_all_bike = ['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt_binary']
columns_all_bike_test = ['workingday','temp', 'cnt_binary', 'hr_new']
create_target_dict_bike = {'column_name_old':['cnt'], 'column_name_new':['cnt_binary'], 'value':[10]}
target_bike = 'cnt_binary'
set_multi_class_bike = ['hr', 6, 12, 18, 24 , 'hr_new']
random_state_bike = 1
training_percent_bike = .08
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
user_optmize_input = ['class', 'constant', 'train', model_score_dict_all]
decision_tree_array_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':[None, 'balanced'], 'random_state':[random_state_bike]}
logistic_regression_array_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1], 'random_state':[random_state_bike]}
neural_net_array_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[True], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08], 'random_state':[random_state_bike]}
database_name = 'machine_predict_test_db'
table_name = 'bike_table1'
db_location_base = '/home/mike/Documents/coding_all/machine_predict/'
bike_predict = MachinePredictModel(df_bike, columns_all_bike_test, random_state_bike, training_percent_bike, kfold_number_bike, target_bike, cols_to_drop=columns_to_drop_bike,set_multi_class=set_multi_class_bike, target_change_bin_dict=create_target_dict_bike, kfold_dict=kfold_dict, param_dict_logistic=logistic_regression_params_bike, param_dict_decision_tree=decision_tree_params_bike, param_dict_neural_network=nnl_params_bike, param_dict_logistic_array=logistic_regression_array_vars, param_dict_decision_tree_array=decision_tree_array_vars, param_dict_neural_network_array=neural_net_array_vars, user_input_for_model_output=user_optmize_input, cycle_vars_user_check=cycle_vars_user_check, minimum_feature_count_for_var_cycle=minimum_feature_count_for_var_cycle, db_location_base=db_location_base, table_name=table_name, database_name= database_name)
data_wanted = bike_predict.user_full_model()

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


# loans model
file_location = '/home/mike/Documents/coding_all/data_sets_machine_predict/cleaned_loans_2007.csv'
df_bike = pd.read_csv(file_location)
#columns_to_drop_bike = ['casual', 'registered', 'dteday']
columns_all_bike = ['loan_amnt', 'int_rate', 'installment', 'emp_length', 'annual_inc',
		'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
	   'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
	   'home_ownership_MORTGAGE', 'home_ownership_NONE',
	   'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
	   'verification_status_Not Verified',
	   'verification_status_Source Verified', 'verification_status_Verified',
	   'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation',
	   'purpose_educational', 'purpose_home_improvement', 'purpose_house',
	   'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
	   'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
	   'purpose_vacation', 'purpose_wedding', 'term_ 36 months',
	   'term_ 60 months', 'loan_status']
columns_all_bike_test = ['loan_amnt', 'int_rate', 'installment', 'emp_length', 
						'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 
						'open_acc', 'loan_status','pub_rec', 'revol_bal', 'revol_util', 'total_acc',
						'home_ownership_MORTGAGE', 'home_ownership_NONE',
						'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
						'verification_status_Not Verified',
						'verification_status_Source Verified', 'verification_status_Verified']
#create_target_dict_bike = {'column_name_old':['cnt'], 'column_name_new':['cnt_binary'], 'value':[10]}
target_bike = 'loan_status'
#set_multi_class_bike = ['hr', 6, 12, 18, 24 , 'hr_new']
random_state_bike = 1
training_percent_bike = .75
kfold_number_bike = 10 
cycle_vars_user_check = 'no'
minimum_feature_count_for_var_cycle = 20
logistic_regression_params_bike = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':'balanced', 'random_state':random_state_bike, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
decision_tree_params_bike = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':random_state_bike, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':'balanced', 'presort':False}
nnl_params_bike = {'hidden_layer_sizes':(10, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':random_state_bike}
kfold_dict = {'n_splits':10, 'random_state':random_state_bike, 'shuffle':False}
model_score_dict_all = {'LogisticRegress':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.85,1], 'fallout_rate':[0,.3]}, 'DecisionTreeCla':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}, 'MLPClassifier(a':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
model_score_dict_nnl = {'MLPClassifier(a':{'roc_auc_score':[.03,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
model_score_dict_log = {'LogisticRegress':{'roc_auc_score':[.55,1], 'precision':[.6,1], 'significant_level':.05, 'sensitivity':[.8,1], 'fallout_rate':[0,.3]}}
model_score_dict_tree = {'DecisionTreeCla':{'roc_auc_score':[.055,1], 'precision':[.06,1], 'significant_level':.05, 'sensitivity':[.06,1], 'fallout_rate':[0,.5]}}
model_score_dict_log_tree = {'DecisionTreeCla':{'roc_auc_score':[.55,1], 'precision':[.6,1], 'significant_level':.05, 'sensitivity':[.6,1], 'fallout_rate':[0,.3]},'LogisticRegress':{'roc_auc_score':[.55,1], 'precision':[.6,1], 'significant_level':.05, 'sensitivity':[.6,1], 'fallout_rate':[0,.3]}}
user_optmize_input = ['class', 'constant', 'train', model_score_dict_all]
decision_tree_array_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':[None, 'balanced'], 'random_state':[random_state_bike]}
logistic_regression_array_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1], 'random_state':[random_state_bike]}
neural_net_array_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[True], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08], 'random_state':[random_state_bike]}
database_name = 'machine_predict_loan_db'
table_name = 'loan_table3'
db_location_base = '/home/mike/Documents/coding_all/machine_predict/'
write_to_db = 'yes'
# sample instance has all vars above in it
loan_instance = MachinePredictModel(df_bike, columns_all_bike, 
				random_state_bike, training_percent_bike, kfold_number_bike, 
				target_bike, 
				kfold_dict=kfold_dict, 
				param_dict_logistic=logistic_regression_params_bike, 
				param_dict_decision_tree=decision_tree_params_bike, 
				param_dict_neural_network=nnl_params_bike, 
				param_dict_logistic_array=logistic_regression_array_vars, 
				param_dict_decision_tree_array=decision_tree_array_vars, 
				param_dict_neural_network_array=neural_net_array_vars, 
				user_input_for_model_output=user_optmize_input, 
				cycle_vars_user_check=cycle_vars_user_check,
				minimum_feature_count_for_var_cycle=minimum_feature_count_for_var_cycle,
				database_name = database_name, table_name = table_name,
				db_location_base = db_location_base, write_to_db = write_to_db)
data = loan_instance.user_full_model()
for k,v in data.items():
	for kk, vv in v.items():
		#print(k)	
		print(kk)
		print(vv)
		print('________________')