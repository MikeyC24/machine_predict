import pandas as pd
import numpy as np
import datetime
import time
from itertools import cycle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc,log_loss, precision_score
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
import operator
from sklearn import preprocessing
# read this of above http://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import math
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# neural network class
# this will be one of the biggest things we relay on
# still much to do here
# the prediction method in this class and the one right below it
# are spitting out different answers then the manual way
# still working on way
class NNet3:

	def __init__(self, learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4, **kwargs):
		self.learning_rate = learning_rate
		self.maxepochs = int(maxepochs)
		self.convergence_thres = 1e-5
		self.hidden_layer = int(hidden_layer)
		#self.random_state = random_state
		self.param_dict = kwargs.get('param_dict_neural_network', None)

	def _sigmoid_activation(self, X, theta):
		X = np.asarray(X)
		theta = np.asarray(theta)
		return 1 / (1 + np.exp(-np.dot(theta.T,X)))
	
	def _multiplecost(self, X, y):
		# feed through network
		l1, l2 = self._feedforward(X) 
		# compute error
		inner = y * np.log(l2) + (1-y) * np.log(1-l2)
		# negative of average error
		return -np.mean(inner)
	
	def _feedforward(self, X):
		# feedforward to the first layer
		l1 = self._sigmoid_activation(X.T, self.theta0).T
		# add a column of ones for bias term
		l1 = np.column_stack([np.ones(l1.shape[0]), l1])
		# activation units are then inputted to the output layer
		l2 = self._sigmoid_activation(l1.T, self.theta1)
		return l1, l2
	
	def predict(self, X):
		_, y = self._feedforward(X)
		return y
	
	def learn(self, X, y):
		nobs, ncols = X.shape
		self.theta0 = np.random.normal(0,0.01,size=(ncols,self.hidden_layer))
		self.theta1 = np.random.normal(0,0.01,size=(self.hidden_layer+1,1))
		
		self.costs = []
		cost = self._multiplecost(X, y)
		self.costs.append(cost)
		costprev = cost + self.convergence_thres+1  # set an inital costprev to past while loop
		counter = 0  # intialize a counter

		# Loop through until convergence
		for counter in range(self.maxepochs):
			# feedforward through network
			l1, l2 = self._feedforward(X)

			# Start Backpropagation
			# Compute gradients
			l2_delta = (y-l2) * l2 * (1-l2)
			l1_delta = l2_delta.T.dot(self.theta1.T) * l1 * (1-l1)

			# Update parameters by averaging gradients and multiplying by the learning rate
			self.theta1 += l1.T.dot(l2_delta.T) / nobs * self.learning_rate
			self.theta0 += X.T.dot(l1_delta)[:,1:] / nobs * self.learning_rate
			
			# Store costs and check for convergence
			counter += 1  # Count
			costprev = cost  # Store prev cost
			cost = self._multiplecost(X, y)  # get next cost
			self.costs.append(cost)
			if np.abs(costprev-cost) < self.convergence_thres and counter > 500:
				break

	# this is giving about a 10% higher answer than by doing the below steps
	# separate not as a class method but with the same variables
	def predict_scores(self, X_train, y_train, X_test, y_test, class_instance):
		model = class_instance
		model.learn(X_train, y_train)
		predictions = model.predict(X_test)[0]
		auc = roc_auc_score(y_test, predictions)
		mse = mean_squared_error(y_test, predictions)
		log_loss_var = log_loss(y_test, predictions)
		#precision_score and accuracy cant handle mix of binary and contineous
		roccurve = fpr, tpr, thresholds = roc_curve(y_test, predictions)
		print(auc, log_loss_var)

	def _create_false_pos_and_false_neg(self, predictions, y_target):
		#df_filter = self.dataframe
		tp_filter = (predictions == 1) & (y_target == 1)
		tn_filter = (predictions == 0) & (y_target == 0)
		fp_filter = (predictions == 1) & (y_target == 0)
		fn_filter = (predictions == 0) & (y_target == 1)
		tp = len(predictions[tp_filter])
		tn = len(predictions[tn_filter])
		fp = len(predictions[fp_filter])
		fn = len(predictions[fn_filter])
		true_positive_rate = tp / (tp+fn)
		false_positive_rate = fp / (fp + tn)
		return true_positive_rate, false_positive_rate

	# neural netowrk from sk learn ( not this has size limitations)
	# lookinto using predict probs here 
	#def neural_learn_sk(self, X_train, y_train, X_test, y_test, activation='logistic', hidden_layer_sizes=100, max_iter=200, alpha =.0001):
		#nnl = MLPClassifier(self.learning_rate, self.random_state, activation=activation, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha)
	def neural_learn_sk(self, X_train, y_train, X_test, y_test):
		if self.param_dict is None:
			print('used default params for neural network')
			self.param_dict = {'hidden_layer_sizes':(100, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':None}
		else:
			print('used user params for neural network')
		nnl = MLPClassifier(hidden_layer_sizes=self.param_dict['hidden_layer_sizes'], activation=self.param_dict['activation'], solver=self.param_dict['solver'], alpha=self.param_dict['alpha'], batch_size=self.param_dict['batch_size'], learning_rate=self.param_dict['learning_rate'], learning_rate_init=self.param_dict['learning_rate_init'], power_t=self.param_dict['power_t'], max_iter=self.param_dict['max_iter'], shuffle=self.param_dict['shuffle'], random_state=self.param_dict['random_state'], tol=self.param_dict['tol'], verbose=self.param_dict['verbose'], warm_start=self.param_dict['warm_start'], momentum=self.param_dict['momentum'], nesterovs_momentum=self.param_dict['nesterovs_momentum'], early_stopping=self.param_dict['early_stopping'], validation_fraction=self.param_dict['validation_fraction'], beta_1=self.param_dict['beta_1'], beta_2=self.param_dict['beta_2'], epsilon=self.param_dict['epsilon'])		
		nnl.fit(X_train, y_train)
		predictions = nnl.predict(X_test)
		tpr_fpr_rates = self._create_false_pos_and_false_neg(predictions, y_test)
		dict ={}
		dict['mse'] = mean_squared_error(y_test, predictions)
		dict['mae'] = mean_absolute_error(y_test, predictions)
		dict['r2_score'] = r2_score(y_test, predictions)
		dict['variance'] = np.var(predictions)
		dict['auc'] = roc_auc_score(y_test, predictions)
		dict['log_loss_var'] = log_loss(y_test, predictions)
		dict['tpr'] = tpr_fpr_rates[0]
		dict['fpr'] = tpr_fpr_rates[1]
		return(dict)


def predict_scores1(X_train, y_train, X_test, y_test, class_instance):
	model = class_instance
	model.learn(X_train, y_train)
	predictions = model.predict(X_test)[0]
	auc = roc_auc_score(y_test, predictions)
	mse = mean_squared_error(y_test, predictions)
	log_loss_var = log_loss(y_test, predictions)
	#precision_score and accuracy cant handle mix of binary and contineous
	roccurve = fpr, tpr, thresholds = roc_curve(y_test, predictions)
	print(auc, log_loss_var)




"""
X = a.dataframe[['ones', 'Transactions_Volume_normalized', 'Number_of_Transactions_normalized', 'LTC_BTC_EX_High_normalized', 'EUR_BTC_EX_High_normalized']].values
y = a.dataframe.target_new.values
# Set a learning rate
learning_rate = 0.5
# Maximum number of iterations for gradient descent
maxepochs = 10000       
# Costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.00001  
# Number of hidden units
hidden_units = 4

# Initialize model 
model = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
			  convergence_thres=convergence_thres, hidden_layer=hidden_units)
# Train model
model.learn(X, y)

# Plot costs
plt.plot(model.costs)
plt.title("Convergence of the Cost Function")
plt.ylabel("J($\Theta$)")
plt.xlabel("Iteration")
plt.show()


X_train = X[:1000]
y_train = y[:1000]
X_test = X[1000:]
y_test = y[1000:]

model = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
			  convergence_thres=convergence_thres, hidden_layer=hidden_units)
model.learn(X_train, y_train)
predictions = model.predict(X_test)[0]
auc = roc_auc_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
log_loss_var = log_loss(y_test, predictions)
#precision_score and accuracy cant handle mix of binary and contineous
#precision_score_var  = precision_score(y_test, predictions)
roccurve = fpr, tpr, thresholds = roc_curve(y_test, predictions)
print(auc, log_loss_var)

target = 'target_new'
var_columns = ['ones', 'Transactions_Volume_normalized', 'Number_of_Transactions_normalized', 'LTC_BTC_EX_High_normalized', 'EUR_BTC_EX_High_normalized']

test_vars = test_project.create_train_and_test_data_x_y_mixer(.07, X, y)
X_train1 = test_vars[0]
y_train1 = test_vars[1]
X_test1 = test_vars[2]
y_test1 = test_vars[3]

model1 = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
			  convergence_thres=convergence_thres, hidden_layer=hidden_units)
model1.learn(X_train1, y_train1)
predictions1 = model1.predict(X_test1)[0]
auc1 = roc_auc_score(y_test1, predictions1)
mse1 = mean_squared_error(y_test1, predictions1)
log_loss_var1 = log_loss(y_test1, predictions1)
#precision_score and accuracy cant handle mix of binary and contineous
#precision_score_var  = precision_score(y_test, predictions)
roccurve = fpr, tpr, thresholds = roc_curve(y_test1, predictions1)
print(auc, log_loss_var)

model.predict_scores(X_train1, y_train1, X_test1, y_test1, model)
predict_scores1(X_train1, y_train1, X_test1, y_test1, model)
"""