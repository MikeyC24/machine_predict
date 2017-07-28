import unittest
from RegressionCombined import *
from ArrangeData import *
from MachinePredictModelrefractored import *

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

"""
# methods to test from regression combined
1. _get_error_scores_with_tpr_fpr
2. classification_unifying_model
3. regression_probs_model_with_user_input
4. regression_probs_model_full_paramter_fit_with_user_input
# methods to test from MachinePredictModel
# 1, 2 and 3 methods will not be tested as their is so much overlap in 
# the arrange data class and bc of the unknown refractoring coming
# also 2 and 3 will be refractored together
1. _set_up_data_for_prob_predict
2. _set_up_data_for_models
3. _set_up_data_for_models_test
4. predict_prob_model_full_fit_parameters(
5. user_full_model
6.return_desired_user_output_from_dict
7. user_output_model
8. cycle_vars_return_desired_output_specific_model
9. _cycle_vars_dict
10. predict_prob_model_full 
"""

# right now these two classes are being tested until 
# better refractoring can be done later
class TestRegressionsAndMachinePredict(unittest.TestCase):

	# this class takes in features, target, kfold_dict, X_train, X_test, y_train, y_test, **kwargs
	# and thus all these will need to be set as clsss vars that can then be fed into the
	# various test to test each function

	# this will take in hour.csv data, these tests are contingent of arrange data working
	# below are the needed vars for the class
	# some of these vars may be added to the actaul methods
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
	# right below should not be here
	#user_params_or_find_params= 'user'
	minimum_feature_count_for_var_cycle = 3
	logistic_regression_params_bike = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':'balanced', 'random_state':random_state_bike, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
	decision_tree_params_bike = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':random_state_bike, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':'balanced', 'presort':False}
	nnl_params_bike = {'hidden_layer_sizes':(10, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':random_state_bike}
	kfold_dict = {'n_splits':10, 'random_state':random_state_bike, 'shuffle':False}
	decision_tree_array_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':[None, 'balanced'], 'random_state':[random_state_bike]}
	logistic_regression_array_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1], 'random_state':[random_state_bike]}
	neural_net_array_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[True], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08], 'random_state':[random_state_bike]}
	# user changeing vars are below, will be adjusted for each case
	#model_score_dict = {'logistic':{'roc_auc_score':.03, 'precision':.06, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}, 'decision_tree':{'error_metric':'roc_auc_score', 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}, 'neural_network':{'error_metric':'roc_auc_score', 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
	#model_score_dict1 = {'neural_network':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
	#model_score_dict2 = {'logistic':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
	#model_score_dict3 = {'decision_tree':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]},'logistic':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
	#user_optmize_input = ['class', 'constant', 'train', model_score_dict3]
	#cycle_vars_user_check = 'no'

	"""
	# differ use cases
	1. cycle or dont cycle vars
	2. user params or not
	3. tree, log, nnl,
	"""
	# case 1 no var cycle user param input, train,  all 6 cases of models
	def test_right_training_method_and_scores(self):
		cycle_vars = 'no'
		model_score_dict_all = {'logistic':{'roc_auc_score':.03, 'precision':.06, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}, 'decision_tree':{'error_metric':'roc_auc_score', 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}, 'neural_network':{'error_metric':'roc_auc_score', 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
		model_score_dict_nnl = {'neural_network':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
		model_score_dict_log = {'logistic':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
		model_score_dict_tree = {'logistic':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
		model_score_dict_tree_log = {'decision_tree':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]},'logistic':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
		model_score_dict_tree_nnl = {'decision_tree':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]},'neural_network':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}	
		model_score_dict_nnl_log = {'neural_network':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]},'logistic':{'roc_auc_score':.03, 'precision':.06, 'significant_level':.05, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
		user_optmize_input = ['class', 'constant', 'train', model_score_dict_all]
		key_poss_of_return_dict = ['dict_results_simple', 'dict_results_kfold', 'dict_results_train_set']
		"""
		bike_predict = MachinePredictModel(self.df_bike, self.columns_all_bike_test, self.random_state_bike, self.training_percent_bike, 
											self.kfold_number_bike, self.target_bike, cols_to_drop=self.columns_to_drop_bike,set_multi_class=self.set_multi_class_bike, 	
											target_change_bin_dict=self.create_target_dict_bike, kfold_dict=self.kfold_dict, 
											param_dict_logistic=logistic_regression_params_bike, param_dict_decision_tree=self.decision_tree_params_bike,
											param_dict_neural_network=self.nnl_params_bike, user_input_for_model_output=user_optmize_input,
											cycle_vars_user_check=cycle_vars) 
		data_wanted = bike_predict.user_full_model()
		for key,value in data_wanted.items():
			print(key)
			print(value)
			print('_________________')
		"""
		# test to make sure right train model was chosen 
		# and there others are blank
		data_choices = ['simple', 'train', 'kfold']
		for x in data_choices:
			model_score_dict_all1 = {'logistic':{'roc_auc_score':.03, 'precision':.06, 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}, 'decision_tree':{'error_metric':'roc_auc_score', 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}, 'neural_network':{'error_metric':'roc_auc_score', 'tpr_range':[.06,1], 'fpr_range':[.0,.05]}}
			user_optmize_input1 = ['class', 'constant', x, model_score_dict_all1]
			if user_optmize_input1[2] == 'train':
				df_bike1 = pd.read_csv(file_location)
				bike_predict = MachinePredictModel(df_bike1, self.columns_all_bike_test, self.random_state_bike, self.training_percent_bike, 
											self.kfold_number_bike, self.target_bike, cols_to_drop=self.columns_to_drop_bike,set_multi_class=self.set_multi_class_bike, 	
											target_change_bin_dict=self.create_target_dict_bike, kfold_dict=self.kfold_dict, 
											param_dict_logistic=logistic_regression_params_bike, param_dict_decision_tree=self.decision_tree_params_bike,
											param_dict_neural_network=self.nnl_params_bike, user_input_for_model_output=user_optmize_input1,
											cycle_vars_user_check=cycle_vars) 
				data_wanted = bike_predict.user_full_model()
				self.assertTrue(len(data_wanted['dict_results_train_set']) > 0)
				self.assertTrue(len(data_wanted['dict_results_kfold']) <= 0)
				self.assertTrue(len(data_wanted['dict_results_simple']) <= 0)
				self.assertTrue(round(data_wanted['dict_results_train_set']['LogisticRegress']['sensitivity'],3) == .829 )
				self.assertTrue(round(data_wanted['dict_results_train_set']['LogisticRegress']['fallout_rate'],3) == .171 )
				self.assertTrue(round(data_wanted['dict_results_train_set']['DecisionTreeCla']['sensitivity'],3) == .842 )
				self.assertTrue(round(data_wanted['dict_results_train_set']['DecisionTreeCla']['fallout_rate'],3) == .158 )
				self.assertTrue(round(data_wanted['dict_results_train_set']['MLPClassifier(a']['sensitivity'],3) == .896 )
				self.assertTrue(round(data_wanted['dict_results_train_set']['MLPClassifier(a']['fallout_rate'],3) == .104 )
			elif user_optmize_input1[2] == 'kfold':
				df_bike1 = pd.read_csv(file_location)
				bike_predict = MachinePredictModel(df_bike1, self.columns_all_bike_test, self.random_state_bike, self.training_percent_bike, 
											self.kfold_number_bike, self.target_bike, cols_to_drop=self.columns_to_drop_bike,set_multi_class=self.set_multi_class_bike, 	
											target_change_bin_dict=self.create_target_dict_bike, kfold_dict=self.kfold_dict, 
											param_dict_logistic=logistic_regression_params_bike, param_dict_decision_tree=self.decision_tree_params_bike,
											param_dict_neural_network=self.nnl_params_bike, user_input_for_model_output=user_optmize_input1,
											cycle_vars_user_check=cycle_vars) 
				data_wanted = bike_predict.user_full_model()
				self.assertTrue(len(data_wanted['dict_results_train_set']) <= 0)
				self.assertTrue(len(data_wanted['dict_results_kfold']) > 0)
				self.assertTrue(len(data_wanted['dict_results_simple']) <= 0)
				self.assertTrue(round(data_wanted['dict_results_kfold']['LogisticRegress']['tpr'], 3) == .830 )
				self.assertTrue(round(data_wanted['dict_results_kfold']['LogisticRegress']['fpr'], 3) == .047 )
				self.assertTrue(round(data_wanted['dict_results_kfold']['DecisionTreeCla']['tpr'], 3) == .796 )
				self.assertTrue(round(data_wanted['dict_results_kfold']['DecisionTreeCla']['fpr'], 3) == .018)
				self.assertTrue(round(data_wanted['dict_results_kfold']['MLPClassifier(a']['tpr'], 3) == .923 )
				self.assertTrue(round(data_wanted['dict_results_kfold']['MLPClassifier(a']['fpr'], 3) == .398)
			elif user_optmize_input1[2] == 'simple':
				df_bike1 = pd.read_csv(file_location)
				bike_predict = MachinePredictModel(df_bike1, self.columns_all_bike_test, self.random_state_bike, self.training_percent_bike, 
											self.kfold_number_bike, self.target_bike, cols_to_drop=self.columns_to_drop_bike,set_multi_class=self.set_multi_class_bike, 	
											target_change_bin_dict=self.create_target_dict_bike, kfold_dict=self.kfold_dict, 
											param_dict_logistic=logistic_regression_params_bike, param_dict_decision_tree=self.decision_tree_params_bike,
											param_dict_neural_network=self.nnl_params_bike, user_input_for_model_output=user_optmize_input1,
											cycle_vars_user_check=cycle_vars) 
				data_wanted = bike_predict.user_full_model()
				self.assertTrue(len(data_wanted['dict_results_train_set']) <= 0)
				self.assertTrue(len(data_wanted['dict_results_kfold']) <= 0)
				self.assertTrue(len(data_wanted['dict_results_simple']) > 0)
				self.assertTrue(round(data_wanted['dict_results_simple']['LogisticRegress']['sensitivity'], 3) == .835)
				self.assertTrue(round(data_wanted['dict_results_simple']['LogisticRegress']['fallout_rate'], 3) == .165)
				self.assertTrue(round(data_wanted['dict_results_simple']['DecisionTreeCla']['sensitivity'], 3) == .833)
				self.assertTrue(round(data_wanted['dict_results_simple']['DecisionTreeCla']['fallout_rate'], 3) == .167)
				self.assertTrue(round(data_wanted['dict_results_simple']['MLPClassifier(a']['sensitivity'], 3) == .970)
				self.assertTrue(round(data_wanted['dict_results_simple']['MLPClassifier(a']['fallout_rate'], 3) == .030)
			else:
				print('improper training method chosen')

		

test_instance = TestRegressionsAndMachinePredict()
test_instance.test_right_training_method_and_scores()

"""
issues
for kfold and train (prob for simple too) when running more than one model
all the nested dicts containing scores are returning the same scores all from one
particular model rather than their wn respective models
***fixed these two issues, havent looked at simple yet but I would still 
call the fix a bandaid at this point as i dont like the consistency 
of the key returned
"""