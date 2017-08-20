from MachinePredictModelrefractored import *
from KerasClass import *

file_location1 = '/home/mike/Documents/coding_all/data_sets_machine_predict/coin_months_data'
file_location = '/home/mike/Downloads/coin_months_data'
#df = pd.read_csv(file_location)
con = sqlite3.connect(file_location)
table1 = 'second_coin_list_two'
table = 'top_3_jan_mid_aug_final'
df = pd.read_sql_query('SELECT * FROM %s' % (table), con)
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
 'amount_USDT_BTC', 'total_USDT_BTC', 'amount_USDT_LTC', 'total_USDT_LTC', 'date',
]
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
rolling_averages_dict = { 'rate_USDT_ETH':[144,288],'rate_USDT_ETH':[144,288],'rate_USDT_ETH':[144,288]}
rolling_std_dict = {'rate_USDT_ETH':[144,288],'rate_USDT_ETH':[144,288],'rate_USDT_ETH':[144,288]}
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

result = sample_instance._set_up_data_for_prob_predict()
df =result.dataframe
print(df.columns.values)
feature_wanted = 'rate_USDT_ETH'
df = df.loc[29425:,]
print(df.shape)


model_type = 'classification'
parameter_type = 'constant'
train_percent = .8
dataframe = df
window = 30
step = 1
forecast = 1
plot = 'yes'
feature_wanted = 'rate_USDT_ETH'
percent_change = 1
"""
keras_instance = KerasClass(model_type, parameter_type, 
	dataframe, window, step, forecast, feature_wanted, train_percent, plot)
#keras_instance.binary_classification_model(1)
#result = keras_instance.create_feature_var_dict()
#X, Y = keras_instance.create_X_Y_values(1)
#X_train, X_test, Y_train, Y_test = keras_instance.create_Xt_Yt(X,Y)
#print(result)
#print(X,Y)
#print(X_train, X_test, Y_train, Y_test)
keras_instance.simple_mlp_example(1)

"""

keras_instance = KerasClass(model_type, parameter_type, 
	dataframe, window, step, forecast, feature_wanted, percent_change,  train_percent, plot)

space ={'window':hp.choice('window', [30]),
		'loss':hp.choice('loss', ['binary_crossentropy', 'categorical_crossentropy']),
		}
space ={'window':hp.choice('window', [6,30,72]),}
#keras_instance.best_params(space)
#keras_instance.binary_classification_model()

# seeing how prepared ata may be written to database
EMB_SIZE = len(df.columns)
X, Y = keras_instance.create_X_Y_values()
X_train, X_test, Y_train, Y_test = keras_instance.create_Xt_Yt(X, Y)
print('X_train shape', X_train.shape, 'xtrain type before reshape',  type(X_train))
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))
y = 1
"""
for x in X_train:
	print(y)
	print(x)
	print(len(x))
	y+=1
"""
try:
	df2 = pd.DataFrame(X_train.reshape(-1,3), columns=list('X_train'))
	print(df2.shape)
	print(df2.columns)
except:
	print('df2 didn;t work')

try:
	df3 = pd.Panel(X_train).to_frame()
	#df3 = pan.swapaxes(0,2).to_frame()
	print(df3.shape)
	print(df3.columns)
	print(df3.head(10))
	print('row 0', df3.iloc[0])
	#x_train_new = df3.as_matrix()
	#print('x train new', x_train_new)
	print('x train', X_train, X_train.shape, 'xtrain_type', type(X_train))
	print('x_train row 0 shape', X_train[0].shape)
	print('df3 shape', df3.shape)
	print('first row shape df3', df3.iloc[0].shape, 'type of df3 first row', type(df3.iloc[0]))
	print('df row 0 without i loc', df3[0])
	#print('trying to reshape first row of df3', '________')
	#print('first converting to numpy nd array', df3[0].values. type(df3[0].values))

	# above shape is 150,30,16
	# panel saves it to dataframe as 480, 150 (16*30 is 480)
	# need to figure out how to unpack panel back to 3d array of 150,30,16
except:
	print('df3 didnt work')

print('trying to reshape first row of df3', '________')
print('first converting to numpy nd array')
df4 =df3.values
print('type of df3.values', type(df4))
print(df4[0])
print('reshaping df4 0')
reshape1 = np.reshape(df4, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
print(reshape1, reshape1.shape)
print('above is reshape, below is x_Train_________')
print('x train', X_train, X_train.shape)
df5 = pd.Panel(X_test).to_frame()
df6 = df5.values
reshape2 = np.reshape(df6, (X_test.shape[0], X_test.shape[1], EMB_SIZE))
print('shape of x train, x test, reshape 1, eshape 2 in order',
	X_train.shape, X_test.shape, reshape1.shape, reshape2.shape)

print('Y_train and y_test shapes', Y_train.shape, Y_test.shape)
y_data_dict = {'y_train':Y_train}
print(type(Y_train))

df_y_train = pd.DataFrame(Y_train)
df_y_test = pd.DataFrame(Y_test)
#df_y_train['y_train'] = Y_train
print(type(df_y_train))
print(df_y_train)
df_y_train_f = np.reshape(df_y_train.values, (150, 2))
print(df_y_train_f)
print('________________________')
df_y_test_f = np.reshape(df_y_test.values, (Y_test.shape[0], 2))
print(df_y_test_f, print(df_y_test_f.shape))


model = Sequential()
model.add(Convolution1D(input_shape = (window, EMB_SIZE),
                        nb_filter=16,
                        filter_length=4,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Convolution1D(nb_filter=8,
                        filter_length=4,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())


model.add(Dense(2))
model.add(Activation('softmax'))

opt = Nadam(lr=0.002)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)


model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(reshape1, df_y_train_f, 
          nb_epoch = 10, 
          batch_size = 128, 
          verbose=1, 
          validation_data=(reshape2, df_y_test_f),
          callbacks=[reduce_lr, checkpointer],
          shuffle=True)

model.load_weights("lolkek.hdf5")
pred = model.predict(np.array(reshape2))


C = confusion_matrix([np.argmax(y) for y in df_y_test_f], [np.argmax(y) for y in pred])
print(C / C.astype(np.float).sum(axis=1))


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


"""
array_check = ['a', 'b', 'c']
df1 = pd.DataFrame()
df1['X_train'] = X_train
df1['Y_train'] = Y_train
df1['X_test'] = X_test
df1['Y_test'] = Y_test
print(df1.shape1)
print(df1.columns)
"""
# https://stackoverflow.com/questions/35525028/how-to-transform-a-3d-arrays-into-a-dataframe-in-python
# https://stackoverflow.com/questions/35525028/how-to-transform-a-3d-arrays-into-a-dataframe-in-python
# set up x,y train and test as columns in df, write df to sql. see if model can just
# load those columns as the vars


"""
print('X_train shape after reshape', X_train.shape)
print('________________')
print('one',X_test[0])
print('two',X_test[0][0])
print('three',X_test[0][0][0])
"""