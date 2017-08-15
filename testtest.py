#https://keras.io/scikit-learn-api/
# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# http://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# https://aboveintelligent.com/time-series-analysis-using-recurrent-neural-networks-lstm-33817fa4c47a
# https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction
# http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
# http://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
# https://datascience.stackexchange.com/questions/17024/rnns-with-multiple-features
# https://www.datacamp.com/community/tutorials/deep-learning-python#gs.obfVGDI

from MachinePredictModelrefractored import *
from DatabaseFunctionality import *
#from keras.models import Sequential
#from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#from utils import *

import pandas as pd
import matplotlib.pylab as plt
"""
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
"""
import seaborn as sns
sns.despine()



file_location = '/home/mike/Documents/coding_all/data_sets_machine_predict/coin_months_data'
#df = pd.read_csv(file_location)
con = sqlite3.connect(file_location)
table = 'second_coin_list_two'
df = pd.read_sql_query('SELECT * FROM %s' % (table), con)
drop_nan_rows = 'yes'
#columns_to_drop = None
#columns_to_drop = ['amount_USDT_ETH','total_USDT_ETH', 'trade_count_USDT_ETH',
#'min_rate_USDT_ETH', 'max_rate_USDT_ETH', 'rate_USDT_ETH', 'rate_USDT_ETH_change', 'date']
columns_to_drop = ['amount_USDT_ETH','total_USDT_ETH',
 'amount_USDT_BTC', 'total_USDT_BTC', 'amount_USDT_LTC', 'total_USDT_LTC', 'date',
 'trade_count_USDT_LTC', 'max_rate_USDT_LTC','rate_USDT_BTC',
 'trade_count_USDT_BTC', 'min_rate_USDT_BTC', 'max_rate_USDT_BTC', 'rate_USDT_LTC',
 'min_rate_USDT_LTC', ]
# columns all before any editing 
columns_all_init = ['date']
# took date out of colums_all
columns_all = [ 'rate_USDT_BTC',  'amount_USDT_BTC',  'total_USDT_BTC', 
'trade_count_USDT_BTC', 'min_rate_USDT_BTC', 'max_rate_USDT_BTC', 'rate_USDT_ETH', 
'rate_USDT_LTC', 'amount_USDT_LTC', 'total_USDT_LTC', 'trade_count_USDT_LTC', 
'max_rate_USDT_LTC', 'max_rate_USDT_LTC', 'date']
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
rolling_averages_dict = { 'rate_USDT_ETH':[6,24]}
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
					rolling_averages_dict=rolling_averages_dict)

result = sample_instance._set_up_data_for_prob_predict()
df =result.dataframe
print(df.columns.values)

#just with ETH vars
#['rate_USDT_ETH' 'trade_count_USDT_ETH' 'min_rate_USDT_ETH'
# 'max_rate_USDT_ETH' 'MA_6_rate_USDT_ETH' 'MA_24_rate_USDT_ETH']

# adjusted change is based on 1 time period aka 10 minutes
rate_USDT_ETH = df.ix[:,'rate_USDT_ETH'].tolist()
trade_count_USDT_ETH = df.ix[:,'trade_count_USDT_ETH'].tolist()
min_rate_USDT_ETH = df.ix[:,'min_rate_USDT_ETH'].tolist()
max_rate_USDT_ETH = df.ix[:,'max_rate_USDT_ETH'].tolist()
MA_6_rate_USDT_ETH = df.ix[:,'MA_6_rate_USDT_ETH'].tolist()
MA_24_rate_USDT_ETH = df.ix[:,'MA_24_rate_USDT_ETH'].tolist()
print(len(rate_USDT_ETH))
print(type(rate_USDT_ETH))

# vars
WINDOW = 30
EMB_SIZE = 6
STEP = 1
FORECAST = 1 
X, Y = [], []
for i in range(0, df.shape[0], STEP): 
	try:
		rate = rate_USDT_ETH[i:i+WINDOW]
		count = trade_count_USDT_ETH[i:i+WINDOW]
		minr = min_rate_USDT_ETH[i:i+WINDOW]
		maxr = max_rate_USDT_ETH[i:i+WINDOW]
		MA6 = MA_6_rate_USDT_ETH[i:i+WINDOW]
		MA24 = MA_24_rate_USDT_ETH[i:i+WINDOW]

		rate = (np.array(rate) - np.mean(rate)) / np.std(rate)
		count = (np.array(count) - np.mean(count)) / np.std(count)
		minr = (np.array(minr) - np.mean(minr)) / np.std(minr)
		maxr = (np.array(maxr) - np.mean(maxr)) / np.std(maxr)
		MA6  = (np.array(MA6) - np.mean(MA6 )) / np.std(MA6)
		MA24  = (np.array(MA24) - np.mean(MA24)) / np.std(MA24)

		x_i = rate_USDT_ETH[i:i+WINDOW]
		y_i = rate_USDT_ETH[i+WINDOW+FORECAST]  
		#print('x_i', x_i)

		last_close = x_i[-1]
		next_close = y_i
		#print('last close', last_close)
		#print('next close', next_close)
		if last_close < next_close:
			y_i = [1, 0]
		else:
			y_i = [0, 1] 
		#print('rate1', rate)
		#print(y_i)
		print('rate', rate)
		print('count', count)
		print('minr', minr)
		print('maxr', maxr)
		print('ma6', MA6)
		print('ma24', MA24)
		x_i = np.column_stack((rate, count, minr, maxr, MA6, MA24))
		print('i', i)
		print('x_i', x_i)
		#print('after stack', x_i)
	except Exception as e:
		#print('hit break')
		break

	X.append(x_i)
	Y.append(y_i)

#print(X)
#print(Y)
#print(len(X))
#print(len(Y))
