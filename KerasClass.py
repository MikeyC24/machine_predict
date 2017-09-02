from MachinePredictModelrefractored import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
#from utils import *
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from keras.models import Sequential
from keras.layers import LSTM
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
from keras import losses
import seaborn as sns
sns.despine()
import sys

# readings https://github.com/FrancisArgnR/Time-series---deep-learning---state-of-the-art

# for now this class is only taking a dataframe from main model that has been
# worked on from the arrange data class, traing/test and all keras model vars
# will be entered into this class until refractored later 

class KerasClass:

	def __init__(self, model_type, parameter_type, dataframe,
		window, step, forecast, feature_wanted, percent_change = 1, train_percent=.8, 
		plot='yes', **kwargs):
		self.model_type = model_type
		self.parameter_type = parameter_type
		self.train_percent = train_percent
		self.dataframe = dataframe
		self.window = window
		self.step = step
		self.forecast = forecast
		self.feature_wanted = feature_wanted
		self.plot = plot
		self.percent_change = percent_change
		self.EMB_SIZE = len(self.dataframe.columns)
		self.write_to_sql = kwargs.get('write_to_sql', None)
		self.read_from_sql_for_model = kwargs.get('read_from_sql_for_model', None)

	def data2change(self, data):
		change = pd.DataFrame(data).pct_change()
		change = change.replace([np.inf, -np.inf], np.nan)
		change = change.fillna(0.).values.tolist()
		change = [c[0] for c in change]
		return change

	def create_feature_var_dict(self):
		df = self.dataframe
		feature_vars_dict = {}
		for column in df.columns.values:
			feature = df.ix[:,column].tolist()
			if self.model_type == 'linear':
				# prob add a percent change step here for linear
				feature = self.data2change(feature)
			feature_vars_dict[str(column)] = feature
		return feature_vars_dict

	def create_X_Y_values(self, window=30):
		print('running create X Y values')
		feature_vars_dict = self.create_feature_var_dict()
		print(len(feature_vars_dict))
		X, Y = [], []
		for i in range(0, self.dataframe.shape[0], self.step):
			print('i', i)
			# for statement here for class or regression
			if self.model_type == 'classification':
				print('buidling data for classificaiton')
				dict_features = {}
				try:
					for feature, feature_data in feature_vars_dict.items():
						# normalize feature this would be cheating for regression
						# but for classification works ok since prediction
						#doesnt need to be exact (cheating bc mean and variance
						# change over time )
						print('feature', feature)
						#print('feature_data', feature_data)
						f = feature_data[i:i+window]
						print('i', i)
						print('window', self.window)
						print('f', f)
						name = str(feature) + '_normalized'
						print('name1', name)
						print('np array', np.array(f))
						print('np mean', np.mean(f))
						print('np std', np.std(f))
						name = (np.array(f) - np.mean(f)) / np.std(f)
						print(name)
						dict_features[str(feature)] = (name)

					# set binary target
					feature_wanted_data = feature_vars_dict[self.feature_wanted]
					x_i = feature_wanted_data[i:i+self.window]
					y_i = feature_wanted_data[i+self.window+self.forecast]
					last_close = x_i[-1]
					next_close = y_i
					if (last_close* self.percent_change) < next_close:
						y_i = [1, 0]
					else:
						y_i = [0, 1]
					# turn x_i into 1d array
					x_i = np.column_stack((dict_features.values()))
					print('x_i', x_i)

				except Exception as e:
					print('hit break')
					print('error: ', e)
					break

				X.append(x_i)
				Y.append(y_i)

			elif self.model_type == 'linear':
				print('building data for linear')
				dict_features = {}
				try:
					for feature, feature_data in feature_vars_dict.items():
						print('feature', feature)
						f = feature_data[i:i+window]
						name = f
						dict_features[str(feature)] =(name)
					feature_wanted_data = feature_vars_dict[self.feature_wanted]
					x_i = np.column_stack((dict_features.values()))
					y_i = y_i = feature_wanted_data[i+self.window+self.forecast]
					print(x_i)
				except Exception as e:
						print('hit break')
						break

				X.append(x_i)
				Y.append(y_i)

			else:
				print('model type not supported ')

		X, Y = np.array(X), np.array(Y)

		return X, Y

	def count_out_ys_by_group(self, values_list, up_value=.02, down_value=-.02):
		up_array = []
		down_array = []
		up_var_array = []
		down_var_array = []
		middle_var_array = []
		for x in values_list:
			up_array.append(x) if x > 0 else down_array.append(x)
		for x in values_list:
			if x > up_value:
				up_var_array.append(x)
			elif x < down_value:
				down_var_array.append(x)
			else:
				middle_var_array.append(x)
		dict_numbers = {}
		dict_numbers['up_values'] = up_array
		dict_numbers['down_values'] = down_array
		dict_numbers['up' +str(up_value) + '_value'] = up_var_array
		dict_numbers['down' +str(down_value) + '_value'] = down_var_array
		dict_numbers['middle' +str(up_value) + str(down_value) + '_value'] = middle_var_array
		return dict_numbers			


	def chunkIt(self, seq, num):
		avg = len(seq) / float(num)
		out = []
		last = 0.0

		while last < len(seq):
			out.append(seq[int(last):int(last + avg)])
			last += avg

		return out

	def separate_dfs_by_cols_even(self, df, num):
		cols = df.columns.values
		col_len = len(df.columns.values)
		chunks = self.chunkIt(cols, num)
		counter = 1
		dfs_array = []
		for array in chunks:
			name = 'df' + str(counter)
			name = df.loc[:,array[0]:array[-1]]
			dfs_array.append(name)
			counter +=1
		return dfs_array

	def write_array_dbs_to_tables(self, df_array, name_var, database):
		conn = sqlite3.connect(database)
		counter = 1
		for df in df_array:
			df.to_sql(name=name_var+str(counter), con=conn, if_exists='fail', index=False)
			counter +=1

	#shuffle training data 
	def shuffle_in_unison(self, a, b):
		# courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
		print('made it to shuffle')
		assert len(a) == len(b)
		shuffled_a = np.empty(a.shape, dtype=a.dtype)
		shuffled_b = np.empty(b.shape, dtype=b.dtype)
		permutation = np.random.permutation(len(a))
		for old_index, new_index in enumerate(permutation):
			shuffled_a[new_index] = a[old_index]
			shuffled_b[new_index] = b[old_index]
		return shuffled_a, shuffled_b

	def create_Xt_Yt(self, X, y):
		print('made it to create train')
		p = int(len(X) * self.train_percent)
		X_train = X[0:p]
		Y_train = y[0:p]
		 
		X_train, Y_train = self.shuffle_in_unison(X_train, Y_train)
	
		X_test = X[p:]
		Y_test = y[p:]
		if self.write_to_sql is not None:
			print('writing to sql')
			try:
				X_array_dfs = []
				conn  = sqlite3.connect(self.write_to_sql['database'])
				df_x_train = pd.Panel(X_train).to_frame()
				df_x_test = pd.Panel(X_test).to_frame()
				df_y_train = pd.DataFrame(Y_train)
				df_y_test = pd.DataFrame(Y_test)
				separated_dfs = self.separate_dfs_by_cols_even(df_x_train, 6)
				self.write_array_dbs_to_tables(separated_dfs, 'x_train', self.write_to_sql['database'])
				separated_dfs = self.separate_dfs_by_cols_even(df_x_test, 6)
				self.write_array_dbs_to_tables(separated_dfs, 'x_test', self.write_to_sql['database'])
				#df_x_train.to_sql(name=self.write_to_sql['x_train'], con=conn, if_exists='fail')
				#df_x_test.to_sql(name=self.write_to_sql['x_test'], con=conn, if_exists='fail')
				df_y_train.to_sql(name=self.write_to_sql['y_train'], con=conn, if_exists='fail')
				df_y_test.to_sql(name=self.write_to_sql['y_test'], con=conn, if_exists='fail')
			except Exception as e:
				print('could not write to sql')
				print('error is ', e)
		return X_train, X_test, Y_train, Y_test

	def read_from_sql_recombine_dfs(self, df_name_array, database):
		conn = sqlite3.connect(database)
		dfs_array = []
		counter = 1
		for name in df_name_array:
			title = 'df_' +str(counter)
			title = pd.read_sql_query('SELECT * FROM %s' % (name), conn)
			dfs_array.append(title)
			counter += 1
		combined_df = pd.concat(dfs_array, axis=1)
		return combined_df


	def binary_classification_model(self):


		# get x,y values, create train/test/set then reshape them
		if self.read_from_sql_for_model is None:
			X, Y = self.create_X_Y_values(self.window)
			X_train, X_test, Y_train, Y_test = self.create_Xt_Yt(X, Y)
			print('X_train shape', X_train.shape)
			X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], self.EMB_SIZE))
			X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], self.EMB_SIZE))
			print('X_train shape after reshape', X_train.shape)
			print(X_train.shape)
			print(X_test.shape)
			print(Y_train.shape)
			print(Y_test.shape)
		else:
			print('using train and test data from sql db')
			con = sqlite3.connect(self.read_from_sql_for_model['database'])
			df_x_train_from_sql = self.read_from_sql_recombine_dfs(self.read_from_sql_for_model['x_train_array'],
				self.read_from_sql_for_model['database'])
			df_x_test_from_sql = self.read_from_sql_recombine_dfs(self.read_from_sql_for_model['x_test_array'],
				self.read_from_sql_for_model['database'])
			df_y_train_from_sql = pd.read_sql_query('SELECT * FROM %s' % (self.read_from_sql_for_model['y_train']), con)
			df_y_test_from_sql = pd.read_sql_query('SELECT * FROM %s' % (self.read_from_sql_for_model['y_test']), con)
			df_y_train_from_sql = df_y_train_from_sql.drop('index', axis=1)
			df_y_test_from_sql = df_y_test_from_sql.drop('index', axis=1)
			X_train = np.reshape(df_x_train_from_sql.values, (df_x_train_from_sql.shape[1], int((df_x_train_from_sql.shape[0])/(self.EMB_SIZE)), self.EMB_SIZE))
			X_test = np.reshape(df_x_test_from_sql.values, (df_x_test_from_sql.shape[1], int((df_x_test_from_sql.shape[0])/(self.EMB_SIZE)),  self.EMB_SIZE))
			if self.model_type == 'classification':
				Y_train = np.reshape(df_y_train_from_sql.values, (df_y_train_from_sql.shape[0], 2))
				Y_test = np.reshape(df_y_test_from_sql.values, (df_y_test_from_sql.shape[0], 2))
			elif self.model_type == 'linear':
				Y_train = df_y_train_from_sql.values
				Y_test = df_y_test_from_sql.values
			else:
				print('no data found for that model type')
			print(' shapes in order', X_train.shape, X_test.shape,
				Y_train.shape, Y_test.shape)


		if self.model_type == 'classification':
			model = Sequential()
			model.add(Convolution1D(input_shape = (self.window, self.EMB_SIZE),
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

			#model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

			model.add(Flatten())

			model.add(Dense(64))
			model.add(BatchNormalization())
			model.add(LeakyReLU())


			model.add(Dense(2))
			model.add(Activation('softmax'))

			opt = Nadam(lr=0.001)

			reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
			checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)


			model.compile(optimizer=opt, 
						  loss='categorical_crossentropy',
						  metrics=['accuracy'])

			history = model.fit(X_train, Y_train, 
					  nb_epoch = 10, 
					  batch_size = 128, 
					  verbose=1, 
					  validation_data=(X_test, Y_test),
					  callbacks=[reduce_lr, checkpointer],
					  shuffle=True)

			model.load_weights("lolkek.hdf5")
			pred = model.predict(np.array(X_test))

		elif self.model_type == 'linear':
			model = Sequential()
			model.add(Convolution1D(input_shape = (self.window, self.EMB_SIZE),
									nb_filter=16,
									filter_length=4,
									border_mode='same'))
			model.add(MaxPooling1D(2))
			model.add(LeakyReLU())
			#model.add(Dropout(0.5))

			model.add(Convolution1D(nb_filter=32,
									filter_length=4,
									border_mode='same'))
			model.add(MaxPooling1D(2))
			model.add(LeakyReLU())
			#model.add(Dropout(0.5))
			model.add(Flatten())

			#model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

			model.add(Dense(16))
			#model.add(BatchNormalization())
			model.add(LeakyReLU())


			model.add(Dense(1))
			model.add(Activation('linear'))

			#opt = Adam(lr=params['lr'])
			#opt = Nadam

			reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
			checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)


			model.compile(optimizer='Nadam', 
						  loss='mean_squared_error')

			history = model.fit(X_train, Y_train, 
					  nb_epoch = 10, 
					  batch_size = 256, 
					  verbose=1, 
					  validation_data=(X_test, Y_test),
					  callbacks=[reduce_lr, checkpointer],
					  shuffle=True)

			model.load_weights("lolkek.hdf5")
			pred = model.predict(np.array(X_test))

		else:
			print('moel type not recongized')

		if self.model_type == 'classification':

			roc = roc_auc_score(Y_test, pred)
			print('ROC: ', roc)
			
			C = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])
			print(C / C.astype(np.float).sum(axis=1))

			if self.plot == 'yes':
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

		elif self.model_type == 'linear':
			original = Y_test
			predicted = pred
			plt.title('Actual(black) and predicted')
			plt.legend(loc='best')
			plt.plot(original, color='black', label = 'Original data')
			plt.plot(predicted, color='blue', label = 'Predicted data')
			plt.show()
			print(np.mean(np.square(predicted - original)))
			print(np.mean(np.abs(predicted - original)))
			print(np.mean(np.abs((original - predicted) / original)))
			check_df = pd.DataFrame()
			original = [float(i) for i in original]
			predicted = [float(i) for i in predicted]
			check_df['actaul'] = original
			check_df['pred'] = predicted
			up_array = []
			down_array = []
			up_2_percent_array = []
			down_2_percent_array = []
			for x in range(check_df.shape[0]):
				if (check_df['pred'].iloc[x]) > check_df['actaul'].iloc[x-1]:
					up_array.append(x)
			for x in range(check_df.shape[0]):
				if (check_df['pred'].iloc[x]) < check_df['actaul'].iloc[x-1]:
					down_array.append(x)
			for x in range(check_df.shape[0]):
				if (check_df['pred'].iloc[x] + .02) > check_df['actaul'].iloc[x-1]:
					up_2_percent_array.append(x)
			for x in range(check_df.shape[0]):
				if (check_df['pred'].iloc[x] - .02) < check_df['actaul'].iloc[x-1]:
					down_2_percent_array.append(x)
			print('up count', len(up_array))
			print('down count', len(down_array))
			print('2 up count', len(up_2_percent_array))
			print('2 down', len(down_2_percent_array))
			pred_group_count = self.count_out_ys_by_group(predicted)
			original_group_count = self.count_out_ys_by_group(original)
			return original_group_count, pred_group_count



#https://github.com/Rachnog/Deep-Trading/blob/master/hyperparameters/hyper.py
# https://github.com/fchollet/keras/issues/1591
	def optimize_experiment_classification(self, params):
		if self.read_from_sql_for_model is None:
			X, Y = self.create_X_Y_values(params['window'])
			X_train, X_test, Y_train, Y_test = self.create_Xt_Yt(X, Y)
			X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], self.EMB_SIZE))
			X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], self.EMB_SIZE))
			window = params['window']
		else:
			print('using train and test data from sql db')
			con = sqlite3.connect(self.read_from_sql_for_model['database'])
			df_x_train_from_sql = self.read_from_sql_recombine_dfs(self.read_from_sql_for_model['x_train_array'],
				self.read_from_sql_for_model['database'])
			df_x_test_from_sql = self.read_from_sql_recombine_dfs(self.read_from_sql_for_model['x_test_array'],
				self.read_from_sql_for_model['database'])
			df_y_train_from_sql = pd.read_sql_query('SELECT * FROM %s' % (self.read_from_sql_for_model['y_train']), con)
			df_y_test_from_sql = pd.read_sql_query('SELECT * FROM %s' % (self.read_from_sql_for_model['y_test']), con)
			df_y_train_from_sql = df_y_train_from_sql.drop('index', axis=1)
			df_y_test_from_sql = df_y_test_from_sql.drop('index', axis=1)
			X_train = np.reshape(df_x_train_from_sql.values, (df_x_train_from_sql.shape[1], int((df_x_train_from_sql.shape[0])/(self.EMB_SIZE)), self.EMB_SIZE))
			X_test = np.reshape(df_x_test_from_sql.values, (df_x_test_from_sql.shape[1], int((df_x_test_from_sql.shape[0])/(self.EMB_SIZE)),  self.EMB_SIZE))
			if self.model_type == 'classification':
				Y_train = np.reshape(df_y_train_from_sql.values, (df_y_train_from_sql.shape[0], 2))
				Y_test = np.reshape(df_y_test_from_sql.values, (df_y_test_from_sql.shape[0], 2))
			elif self.model_type == 'linear':
				Y_train = df_y_train_from_sql.values
				Y_test = df_y_test_from_sql.values
			else:
				print('no data found for that model type')
			print(' shapes in order', X_train.shape, X_test.shape,
				Y_train.shape, Y_test.shape)
			window = self.window
		#try: 
		if self.model_type == 'classification':
			print('params set up for classificaion optimize')
			try:
				model = Sequential()
				model.add(Convolution1D(input_shape = (window, self.EMB_SIZE),
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
				model.add(Activation(params['activation']))

				opt_use = params['optimizer']
				print(opt_use)
				#opt = opt_use(lr=0.001)

				reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
				checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)


				model.compile(optimizer=opt_use, 
							  loss='categorical_crossentropy',
							  metrics=['accuracy'])

				history = model.fit(X_train, Y_train, 
						  nb_epoch = 10, 
						  batch_size = 128, 
						  verbose=1, 
						  validation_data=(X_test, Y_test),
						  callbacks=[reduce_lr, checkpointer],
						  shuffle=True)

				model.load_weights("lolkek.hdf5")
				pred = model.predict(np.array(X_test))
				acc = roc_auc_score(Y_test, pred)
				print('AUC: ', acc)
				#loss = losses.categorical_crossentropy(Y_test, pred)
				#print('loss: ', loss)

			except Exception as e:
				print('got error: ', e)
				return {'loss':9999999, 'status':STATUS_OK}

	
			#C = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])
			#print('c', C)
			#print(C / C.astype(np.float).sum(axis=1))
			sys.stdout.flush()
			return {'loss':-acc, 'status':STATUS_OK}

		elif self.model_type == 'linear':
			print('setting up params for linear optimize')
			try:
				model = Sequential()
				model.add(Convolution1D(input_shape = (self.window, self.EMB_SIZE),
										nb_filter=16,
										filter_length=4,
										border_mode='same'))
				model.add(MaxPooling1D(2))
				model.add(LeakyReLU())
				#model.add(Dropout(0.5))

				model.add(Convolution1D(nb_filter=32,
										filter_length=4,
										border_mode='same'))
				model.add(MaxPooling1D(2))
				model.add(LeakyReLU())
				#model.add(Dropout(0.5))
				model.add(Flatten())

				#model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

				model.add(Dense(16))
				#model.add(BatchNormalization())
				model.add(LeakyReLU())


				model.add(Dense(1))
				model.add(Activation(params['activation']))

				opt = params['optimizer']

				reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
				checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)


				model.compile(optimizer=opt, 
							  loss=params['loss'])

				history = model.fit(X_train, Y_train, 
						  nb_epoch = 100, 
						  batch_size = 256, 
						  verbose=1, 
						  validation_data=(X_test, Y_test),
						  callbacks=[reduce_lr, checkpointer],
						  shuffle=True)

				model.load_weights("lolkek.hdf5")
				pred = model.predict(np.array(X_test))
			except Exception as e:
				print('something happened, error is : ', e)
				return {'loss':9999999, 'status':STATUS_OK}


			mse = np.mean(np.square(pred - Y_test))    

			if np.isnan(mse):
				print('NaN happened')
				print('-' * 10)
				return {'loss': 999999, 'status': STATUS_OK}

			print(mse)
			print('-' * 10)

			sys.stdout.flush() 
			return {'loss': mse, 'status': STATUS_OK}

		else:
			print('model type not recongized')

	def best_params(self, space):

		trials = Trials()
		best = fmin(self.optimize_experiment_classification, space, algo=tpe.suggest,
			max_evals=5, trials=trials)
		print('best: ')
		print(best)


"""
1.
model predict v evaluate 
https://github.com/fchollet/keras/issues/5140
2.
hyperopt wrapper for keras
https://github.com/maxpumperla/hyperas
3. another time series example
https://gist.github.com/jkleint/1d878d0401b28b281eb75016ed29f2ee
4. more on keras vars 
https://github.com/fchollet/keras/issues/2483
5. for getting amount of percent change 
https://github.com/Rachnog/Deep-Trading/blob/master/volatility/volatility.py#L133
6. battle overfitting a. make sure no data leak b. overfitting, find ways to reduce this
3. kfold
7. more hyper opt and keras github
https://github.com/fchollet/keras/issues/1591
"""

"""
road map in no order
1. linear regression - seems to be working and generating values/predictions
2. kfold option
3. ways to reduce over fitting
4. param optimize for class - getting values from this but dont know if they are right
5. param optimize for linear - getting values from this but they seem to be wrong
other things to look into
6. for rolling avg and std, do rows have to shift back to match window
7. look into lstm and also shifting time series data
links - https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/ 
for saving to sql purposes, this way may be better, shift data before hand
https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
8. trading simulation, be able to keep indexes attached to predictino and apply
logic to trad such up 2% make trade if not in, else stay in, down 2% exit,
then find a way to run back tests
9. ltsm multivarate model, start with one layer, then way to add layer optionally with hyperpam

"""