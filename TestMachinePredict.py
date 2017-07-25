from MachinePredictModelrefractored import *
import unittest

# test case here is from hour.csv which has real bike usage data
# this is first test case wirtten, so bear with it

# all user entered varaibles 
# file location and loading it to df
file_location = '/home/mike/Documents/coding_all/machine_predict/hour.csv'
df_bike = pd.read_csv(file_location)
# cols to drop and use
columns_to_drop_bike = ['casual', 'registered', 'dtedat']
columns_all_bike_test = ['workingday','temp', 'cnt_binary', 'hr_new']
# creating target into binary measure and name of target 
create_target_dict_bike = {'column_name_old':['cnt'], 'column_name_new':['cnt_binary'], 'value':[10]}
target_bike = 'cnt_binary'
# this set needs to make on the columns a multi class
set_multi_class_bike = ['hr', 6, 12, 18, 24 , 'hr_new']
# training figures
random_state_bike = 1
training_percent_bike = .08
kfold_number_bike = 10 
# user entered params for logistic 
logistic_regression_params_bike = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':'balanced', 'random_state':random_state_bike, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
# user entered params for decision tree
decision_tree_params_bike = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':random_state_bike, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':'balanced', 'presort':False}
#user entered params for nnl
nnl_params_bike = {'hidden_layer_sizes':(10, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':random_state_bike}
# kfold dict if kfold method is chosen
kfold_dict = {'n_splits':10, 'random_state':random_state_bike, 'shuffle':False}
# decision tree vars for optimize params
decision_tree_array_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':[None, 'balanced'], 'random_state':[random_state_bike]}
# logistic vars for optimize params
logistic_regression_array_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1], 'random_state':[random_state_bike]}
# nnl vars for optimize params
neural_net_array_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[True], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08], 'random_state':[random_state_bike]}
# not sure where to put the below vars as they will need to change for some of the tests
cycle_vars_user_check = 'no'
user_params_or_find_params= 'user'
model_score_dict_all = {'logistic':{'roc_auc_score':.03, 'precision':.06, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}, 'decision_tree':{'error_metric':'roc_auc_score', 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}, 'neural_network':{'error_metric':'roc_auc_score', 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
model_score_dict_nnl = {'neural_network':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
model_score_dict_log = {'logistic':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
model_score_dict_tree = {'logistic':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
model_score_dict_tree_log = {'decision_tree':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]},'logistic':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
date_unix = 'dteday'
# first testing will involve varaibles below, although some will change dramatically
# to check it data is being set up right

# this thre an error for the format_unix_date, i think i need to test
# arrange data first
user_optmize_input = ['class', 'constant', 'train', model_score_dict_log]
cycle_vars_user_check = 'no'
user_params_or_find_params= 'user'
bike_predict = MachinePredictModel(df_bike, columns_all_bike_test, random_state_bike, training_percent_bike, kfold_number_bike, target_bike, cols_to_drop=columns_to_drop_bike, date_unix = date_unix, set_multi_class=set_multi_class_bike, target_change_bin_dict=create_target_dict_bike, kfold_dict=kfold_dict, param_dict_logistic=logistic_regression_params_bike, param_dict_decision_tree=decision_tree_params_bike, param_dict_neural_network=nnl_params_bike, param_dict_logistic_array=logistic_regression_array_vars, param_dict_decision_tree_array=decision_tree_array_vars, param_dict_neural_network_array=neural_net_array_vars, user_input_for_model_output=user_optmize_input, cycle_vars_user_check=cycle_vars_user_check, minimum_feature_count_for_var_cycle=minimum_feature_count_for_var_cycle)
def test_check_datetime_conversation_works():
	# class instance of MachinePredictModel
	print(type(bike_predict))
	data = bike_predict._set_up_data_for_prob_predict()
	print(type(data))


test_check_datetime_conversation_works()
