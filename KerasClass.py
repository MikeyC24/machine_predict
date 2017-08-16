from MachinePredictModelrefractored import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#from utils import *
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
"""
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
import seaborn as sns
sns.despine()
"""
# for now this class is only taking a dataframe from main model that has been
# worked on from the arrange data class, traing/test and all keras model vars
# will be entered into this class until refractored later 

class KerasClass:

	def __init__(self, model_type, parameter_type, dataframe,
		window, step, forecast,train_percent=.8, plot='yes'):
		self.model_type = model_type
		self.parameter_type = parameter_type
		self.train_percent = train_percent
		self.dataframe = dataframe
		self.window = window
		self.step = step
		self.forecast = forecast
		self.plot = plot
		self.EMB_SIZE = len(self.dataframe.columns)

	def create_feature_var_dict(self):
		df = self.dataframe
		feature_vars_dict = {}
		for column in df.columns.values:
			feature = df.ix[:,column].tolist()
			feature_vars_dict[str(column)] = feature
		EMB = (feature_vars_dict)
		return feature_vars_dict

	def create_X_Y_values(self, change_percent=1):
		feature_vars_dict = self.create_feature_var_dict()
		X, Y = [], []
		for i in range(0, self.dataframe.shape[0], self.step):
			dict_features = {}
			try:
				for feature, feature_data in feature_vars_dict.items():
					# normalize feature this would be cheating for regression
					# but for classification works ok (cheating bc mean and variance
					# change over time )
					print('feature', feature)
					f = feature_data[i:i+WINDOW]
					name = str(feature) + '_normalized'
					name = (np.array(f) - np.mean(f)) / np.std(f)
					dict_features[str(feature)] = (name)

				# set binary target
				feature_wanted_data = feature_vars_dict[feature_wanted]
				x_i = feature_wanted_data[i:i+WINDOW]
				y_i = feature_wanted_data[i+WINDOW+FORECAST]
				last_close = x_i[-1]
				next_close = y_i
				if (last_close*change_percent) < next_close:
					y_i = [1, 0]
				else:
					y_i = [0, 1]
				# turn x_i into 1d array
				x_i = np.column_stack((dict_features.values()))
				print('x_i', x_i)

			except Exception as e:
				print('hit break')
				break

			X.append(x_i)
			Y.append(y_i)

		return X, Y

	#shuffle training data 
	def shuffle_in_unison(self, a, b):
		# courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
		assert len(a) == len(b)
		shuffled_a = np.empty(a.shape, dtype=a.dtype)
		shuffled_b = np.empty(b.shape, dtype=b.dtype)
		permutation = np.random.permutation(len(a))
		for old_index, new_index in enumerate(permutation):
			shuffled_a[new_index] = a[old_index]
			shuffled_b[new_index] = b[old_index]
		return shuffled_a, shuffled_b

	def create_Xt_Yt(self, X, y):
		p = int(len(X) * self.train_percent)
		X_train = X[0:p]
		Y_train = y[0:p]
		 
		X_train, Y_train = self.shuffle_in_unison(X_train, Y_train)
	
		X_test = X[p:]
		Y_test = y[p:]

		return X_train, X_test, Y_train, Y_test

	def binary_classification_model(self, change_percent=1):

		# get x,y values, create train/test/set then reshape them
		X, Y = self.create_X_Y_values(change_percent)
		X_train, X_test, Y_train, Y_test = self.create_Xt_Yt(X, Y)
		X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], self.EMB_SIZE))
		X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], self.EMB_SIZE))

		model = Sequential()
		model.add(Convolution1D(input_shape = (WINDOW, self.EMB_SIZE),
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

		history = model.fit(X_train, Y_train, 
		          nb_epoch = 100, 
		          batch_size = 128, 
		          verbose=1, 
		          validation_data=(X_test, Y_test),
		          callbacks=[reduce_lr, checkpointer],
		          shuffle=True)

		model.load_weights("lolkek.hdf5")
		pred = model.predict(np.array(X_test))

		from sklearn.metrics import classification_report
		from sklearn.metrics import confusion_matrix
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

