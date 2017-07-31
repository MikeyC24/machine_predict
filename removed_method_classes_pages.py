# this page has everything not being used as (except from arrange data) being pulled out until
# it is added back later in a more organized fashion

#from RegressionCombined.py
# two methods

# the simple parameters should prob use train data instead
	def regression_probs_model(self, **kwargs):
		param_dict_logistic = kwargs.get('param_dict_logistic', None)
		param_dict_decision_tree = kwargs.get('param_dict_decision_tree', None)
		param_dict_neural_network = kwargs.get('param_dict_neural_network', None)
		kfold = KFold(self.features.shape[0], n_folds=self.kfold_dict['n_splits'],random_state=self.kfold_dict['random_state'],shuffle=self.kfold_dict['shuffle'])
		# look into if these if else stamtents can be turned into one line
		if param_dict_logistic is None:
			print('used default params for logistic regression')
			param_dict_logistic = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':None, 'random_state':None, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
		else:
			print('used user params for logistic regression')
		if param_dict_decision_tree is None:
			print('used default params for decision tree')
			param_dict_decision_tree = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':None, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':None, 'presort':False}
		else:
			print('used user params for decision tree')
		if param_dict_neural_network is None:
			print('used default params for neural network')
			param_dict_neural_network = {'hidden_layer_sizes':(100, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':None}
		else:
			print('used user params for neural network')
		#print(param_dict_logistic)
		dict_results_kfold = {}
		dict_results_simple = {}
		dict_results_train_set = {}
		dict_all = {}
		reg = LogisticRegression(penalty=param_dict_logistic['penalty'], dual=param_dict_logistic['dual'], tol=param_dict_logistic['tol'], C=param_dict_logistic['C'], fit_intercept=param_dict_logistic['fit_intercept'], intercept_scaling=param_dict_logistic['intercept_scaling'], class_weight=param_dict_logistic['class_weight'], random_state=param_dict_logistic['random_state'], solver=param_dict_logistic['solver'], max_iter=param_dict_logistic['max_iter'], multi_class=param_dict_logistic['multi_class'], verbose=param_dict_logistic['verbose'], warm_start=param_dict_logistic['warm_start'], n_jobs=param_dict_logistic['n_jobs'])
		tree = DecisionTreeClassifier(criterion=param_dict_decision_tree['criterion'], splitter=param_dict_decision_tree['splitter'], max_depth=param_dict_decision_tree['max_depth'], min_samples_split=param_dict_decision_tree['min_samples_split'], min_samples_leaf=param_dict_decision_tree['min_samples_leaf'], min_weight_fraction_leaf=param_dict_decision_tree['min_weight_fraction_leaf'], max_features=param_dict_decision_tree['max_features'], random_state=param_dict_decision_tree['random_state'], max_leaf_nodes=param_dict_decision_tree['max_leaf_nodes'], min_impurity_split=param_dict_decision_tree['min_impurity_split'], class_weight=param_dict_decision_tree['class_weight'], presort=param_dict_decision_tree['presort'])
		nnl = MLPClassifier(hidden_layer_sizes=param_dict_neural_network['hidden_layer_sizes'], activation=param_dict_neural_network['activation'], solver=param_dict_neural_network['solver'], alpha=param_dict_neural_network['alpha'], batch_size=param_dict_neural_network['batch_size'], learning_rate=param_dict_neural_network['learning_rate'], learning_rate_init=param_dict_neural_network['learning_rate_init'], power_t=param_dict_neural_network['power_t'], max_iter=param_dict_neural_network['max_iter'], shuffle=param_dict_neural_network['shuffle'], random_state=param_dict_neural_network['random_state'], tol=param_dict_neural_network['tol'], verbose=param_dict_neural_network['verbose'], warm_start=param_dict_neural_network['warm_start'], momentum=param_dict_neural_network['momentum'], nesterovs_momentum=param_dict_neural_network['nesterovs_momentum'], early_stopping=param_dict_neural_network['early_stopping'], validation_fraction=param_dict_neural_network['validation_fraction'], beta_1=param_dict_neural_network['beta_1'], beta_2=param_dict_neural_network['beta_2'], epsilon=param_dict_neural_network['epsilon'])
		instance_array = [reg, tree, nnl]
		instance_array_name = ['reg_model', 'tree_model', 'nnl_model']
		#reg_instance = reg.fit(self.features, self.target)
		for x in instance_array:
			for y in range(len(instance_array)):
				instance = x.fit(self.features, self.target)
				predictions = x.predict(self.features)
				results = self._get_error_scores_with_tpr_fpr(self.target, predictions)
				dict_results_simple[instance_array_name[y]] = results
		#return dict_results_simple
		# train set iteration
		for x in instance_array:
			for y in range(len(instance_array)):
				instance = x.fit(self.X_train, self.y_train)
				predictions = x.predict(self.X_test)
				results = self._get_error_scores_with_tpr_fpr(self.y_test, predictions)
				dict_results_train_set[instance_array_name[y]] = results
		 
		for x in instance_array:
			for y in range(len(instance_array)):
				dict ={}
				variance_values = []
				mse_values = []
				ame_values =[]
				r2_score_values = []
				true_positive_rate_values = []
				false_positive_rate_values = []
				for train_index, test_index in kfold:
					X_train, X_test = self.features.iloc[train_index], self.features.iloc[test_index]
					y_train, y_test = self.target.iloc[train_index], self.target.iloc[test_index]
					instance = x.fit(self.features, self.target)
					predictions = x.predict(X_test)
					mse = mean_squared_error(y_test, predictions)
					variance = np.var(predictions)
					mae = mean_absolute_error(y_test, predictions)
					r2_scores = r2_score(y_test, predictions)
					#append to array 
					variance_values.append(variance)
					mse_values.append(mse)
					ame_values.append(mae)
					r2_score_values.append(r2_scores)
					tp_filter = (predictions == 1) & (y_test == 1)
					tn_filter = (predictions == 0) & (y_test == 0)
					fp_filter = (predictions == 1) & (y_test == 0)
					fn_filter = (predictions == 0) & (y_test == 1)
					tp = len(predictions[tp_filter])
					tn = len(predictions[tn_filter])
					fp = len(predictions[fp_filter])
					fn = len(predictions[fn_filter])
					true_positive_rate = tp / (tp+fn)
					false_positive_rate = fp / (fp + tn)
					true_positive_rate_values.append(true_positive_rate)
					false_positive_rate_values.append(false_positive_rate)
				dict['avg_mse'] = np.mean(mse_values)
				dict['avg_ame'] = np.mean(ame_values)
				dict['r2_score_values'] = np.mean(r2_score_values)
				dict['ave_var'] = np.mean(variance_values)
				dict['tpr'] = np.mean(true_positive_rate)
				dict['fpr'] = np.mean(false_positive_rate)
				dict_results_kfold[instance_array_name[y]] = dict
		dict_all['dict_results_simple'] = dict_results_simple
		dict_all['dict_results_kfold'] = dict_results_kfold
		dict_all['dict_results_train_set'] = dict_results_train_set
		return dict_all

	def regression_probs_model_full_paramter_fit(self, **kwargs):
		param_dict_logistic_array = kwargs.get('param_dict_logistic_array', None)
		param_dict_decision_tree_array = kwargs.get('param_dict_decision_tree_array', None)
		param_dict_neural_network_array = kwargs.get('param_dict_neural_network_array', None)
		dtree = DecisionTreeClassifier()
		reg = LogisticRegression()
		nnl = MLPClassifier()
		dict_results_parameter_fit = {}
		if param_dict_logistic_array is not None:
			print('doing log regress')
			clf = GridSearchCV(reg, param_dict_logistic_array)
			clf.fit(self.X_train, self.y_train)
			predictions = clf.predict(self.X_test)
			error_score = self._get_error_scores_with_tpr_fpr(self.y_test, predictions)
			error_score['best_Score'] = clf.best_score_
			error_score['best_params'] = clf.best_params_
			dict_results_parameter_fit['logsitic'] = error_score
		if param_dict_decision_tree_array is not None:
			print('doing decision tree')
			clf = GridSearchCV(dtree, param_dict_decision_tree_array)
			clf.fit(self.X_train, self.y_train)
			predictions = clf.predict(self.X_test)
			error_score = self._get_error_scores_with_tpr_fpr(self.y_test, predictions)
			error_score['best_Score'] = clf.best_score_
			error_score['best_params'] = clf.best_params_
			dict_results_parameter_fit['decision_tree'] = error_score
		if param_dict_neural_network_array is not None:
			print('doing nnl')
			clf = GridSearchCV(nnl, param_dict_neural_network_array)
			clf.fit(self.X_train, self.y_train)
			predictions = clf.predict(self.X_test)
			error_score = self._get_error_scores_with_tpr_fpr(self.y_test, predictions)
			error_score['best_Score'] = clf.best_score_
			error_score['best_params'] = clf.best_params_
			dict_results_parameter_fit['neural_net'] = error_score
		return dict_results_parameter_fit

# from machinepredictmodelrefractored 
# just one method

	# iterate over self.columns_all to return different combinations of columns_all
	# to run models on
	# i dont think the is being used
	def _cycle_vars(self):
		cols_array = []
		cols = self.columns_all
		combos_array = []
		dict = {}
		y = 0
		for x in range(0, len(cols)+1):
			for subset in itertools.combinations(cols, x):
				#print(subset)
				combos_array.append(subset)
				dict[y] = subset
				y +=1 	
		return dict

# clustering.py whoe class

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

# this will be the clustering class, aka unsupervised machine learning
# idea of this class is two show how rows of data are related
# all the methods will return values, but for some the values are wrong
class Clustering:

	def __init__(self, dataframe, var1, var2, column_name, cluster_num=5):
		self.cluster_num = cluster_num
		self.dataframe = dataframe
		self.var1 = var1
		self.var2 = var2
		self.column_name = column_name

	# needs to be renamed
	def clustering_two_var_scatter(self):
		df = self.dataframe
		var1= self.var1
		var2= self.var2
		num_clusters = self.cluster_num
		random_initial_points = np.random.choice(df.index, size=num_clusters)
		centroids = df.loc[random_initial_points]
		# plot of all data and the random ones highlighted a diff color, in this case red
		# this below needs to bne turned into a graph method
		"""
		plt.scatter(df[var1], df[var2], c='yellow')
		plt.scatter(centroids[var1], centroids[var2], c='red')
		plt.title("Centroids")
		plt.xlabel(var1, fontsize=13)
		plt.ylabel(var2, fontsize=13)
		plt.show()
		"""
		return centroids

	def recalculate_centroids_dict(self, column_name):
		df = self.dataframe
		new_centroids_dict = dict()
		for cluster_id in range(0, self.cluster_num):
			values_in_cluster = df[df[column_name] == cluster_id]
			#calculate mean of new centroid
			new_centroid = [np.average(values_in_cluster[self.var1]), np.average(values_in_cluster[self.var2])]
			new_centroids_dict[cluster_id] = new_centroid
		return new_centroids_dict

	# from that centroid, create a dict with the id as the key and the two metrics as coordinates
	def centroids_to_dict(self):
		centroids = self.clustering_two_var_scatter()
		dictionary = dict()
		# iterating counter we use to generate a cluster_id
		counter = 0
		# iterate a pandas data frame row-wise using .iterrows()
		for index, row in centroids.iterrows():
			coordinates = [row[self.var1], row[self.var2]]
			dictionary[counter] = coordinates
			counter += 1

		return dictionary

    # using prior two methods, making a new column that assigns each row to a cluster
    # eclidean_distance giving me a dimension
    # this equations only gives one point, need to find a way
    # thru method to iterate down whole row
	def assign_to_cluster(self):
	    lowest_distance = -1
	    closest_distance = -1
	    df = self.dataframe
	    #centroids = self.clustering_two_var_scatter()
	    #closest_cluster_array =[]
	    #centroids_dict = self.centroids_to_dict()
	    centroids_dict = self.centroids_to_dict()
	    print(centroids_dict)	    
	    # centroids_dict is results from centroids_to_dict method
	    # key is counter, vaulue is coordinate
	    for cluster_id, centroid in centroids_dict.items():
	        #a = df[df[var1 == var1]]
	        #b = df[df[var2 == var2]]
	        df_row = [df[self.var1][0], df[self.var2][0]]
	        #df_row = [a,b]
	        #print(df)
	        #print(df[var1])
	        #print(len(df_row))
	        #print(len(centroid))
	        #print(type(df_row))
	        #print(type(centroid))
	        #print(df_row)
	        #print(centroid)
	        
	        # calculate distance is ethod to give distance of two coordinates
	        # right now we are caluclating each players distance from the centroid 
	        euclidean_distance = euclidean_distances(centroid, df_row)
	        #euclidean_distance = self._calculate_distance(centroid, df_row)
	        # once we have distance we want to return which centroid is closest
	        if lowest_distance == -1:
	            lowest_distance = euclidean_distance
	            closest_cluster = cluster_id
	            #print(closest_cluster)
	            #closest_cluster_array.append(closest_cluster)
	            #return closest_cluster
	            #df['cluster'] = closest_cluster 
	        elif euclidean_distance < lowest_distance:
	            lowest_distance = euclidean_distance
	            closest_cluster = cluster_id
	            #return clostest cluster
	            #closest_cluster_array.append(closest_cluster)
	            #print(closest_cluster)
	            #df['cluster'] = closest_cluster
	    return closest_cluster

	def assign_to_cluster_recalc(self):
	    lowest_distance = -1
	    closest_distance = -1
	    df = self.dataframe
	    #centroids = self.clustering_two_var_scatter()
	    #closest_cluster_array =[]
	    #centroids_dict = self.centroids_to_dict()
	    centroids_dict = self.recalculate_centroids_dict(self.column_name)    
	    print(centroids_dict)
	    # centroids_dict is results from centroids_to_dict method
	    # key is counter, vaulue is coordinate
	    for cluster_id, centroid in centroids_dict.items():
	        #a = df[df[var1 == var1]]
	        #b = df[df[var2 == var2]]
	        df_row = [df[self.var1][0], df[self.var2][0]]
	        #df_row = [a,b]
	        #print(df)
	        #print(df[var1])
	        #print(len(df_row))
	        #print(len(centroid))
	        #print(type(df_row))
	        #print(type(centroid))
	        #print(df_row)
	        #print(centroid)
	        
	        # calculate distance is ethod to give distance of two coordinates
	        # right now we are caluclating each players distance from the centroid 
	        euclidean_distance = euclidean_distances(centroid, df_row)
	        #euclidean_distance = self._calculate_distance(centroid, df_row)
	        # once we have distance we want to return which centroid is closest
	        if lowest_distance == -1:
	            lowest_distance = euclidean_distance
	            closest_cluster = cluster_id
	            #print(closest_cluster)
	            #closest_cluster_array.append(closest_cluster)
	            #return closest_cluster
	            #df['cluster'] = closest_cluster 
	        elif euclidean_distance < lowest_distance:
	            lowest_distance = euclidean_distance
	            closest_cluster = cluster_id
	            #return clostest cluster
	            #closest_cluster_array.append(closest_cluster)
	            #print(closest_cluster)
	            #df['cluster'] = closest_cluster
	    return closest_cluster

	# the second recalc is not working
	# the first df is just the random points clustered, still
	# needs to be better refined after that hance the new_dict_recalc
	# method 
	def apply_cluster_to_new_column(self):
		#first_dict = self.centroids_to_dict()
		df =self.dataframe
		column_name = self.column_name
		df[column_name] = df.apply(lambda x: self.assign_to_cluster(), axis=1)
		#second_dict = self.recalculate_centroids_dict(column_name)
		#df[column_name] = df.apply(lambda x: self.assign_to_cluster_recalc(), axis=1)
		return df

	def apply_cluster_to_new_column_recalc(self):
		#first_dict = self.centroids_to_dict()
		df =self.dataframe
		column_name = self.column_name
		df = self.apply_cluster_to_new_column()
		#second_dict = self.recalculate_centroids_dict(column_name)
		df[column_name] = df.apply(lambda x: self.assign_to_cluster_recalc(), axis=1)
		return df


	def visualize_clusters(self, dataframe, var1, var2):
		colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
		column_name = self.column_name
		df = dataframe
		num_clusters = self.cluster_num
		print(num_clusters)
		print(type(num_clusters))
		for x in range(num_clusters):
			clustered_df = df[df[column_name] == x]
			plt.scatter(clustered_df[var1], clustered_df[var2],c=colors[x-1])
			plt.xlabel(var1, fontsize=12)
			plt.ylabel(var2, fontsize=12)
		plt.show()
	    #df['cluster'] = df.apply(lambda row: assign_to_cluster(row), axis=1)
	    #return df

# clustering class work start
dataframe_var = test_project.dataframe
clustering_instance = Clustering(dataframe=dataframe_var, var1='EUR_BTC_EX_High', var2='Transactions_Volume', column_name='cluster')
#clustering_instance.clustering_two_var_scatter()
#a = clustering_instance.centroids_to_dict()
#print(a)
#b = clustering_instance.assign_to_cluster()
#print(b)
#c = clustering_instance.apply_cluster_to_new_column()
#print(type(c))
#print(c.head(25))
#clustering_instance.visualize_clusters(c, 'EUR_BTC_EX_High', 'Transactions_Volume')
c = clustering_instance.apply_cluster_to_new_column()
clustering_instance.visualize_clusters(c, 'EUR_BTC_EX_High', 'Transactions_Volume')
d = clustering_instance.apply_cluster_to_new_column_recalc()
print(d.head(25))
clustering_instance.visualize_clusters(d, 'EUR_BTC_EX_High', 'Transactions_Volume')


#centroids_var = clustering_instance.clustering_two_var_scatter(dataframe_var,'EUR_BTC_EX_High', 'Transactions_Volume')
#centroids_dict = clustering_instance.centroids_to_dict(centroids_var, 'EUR_BTC_EX_High', 'Transactions_Volume')
#print(centroids_dict)
#check = clustering_instance.assign_to_cluster(centroids_var, 'EUR_BTC_EX_High', 'Transactions_Volume', dataframe_var)
#print(check)

#clustering_instance.visualize_clusters(check, 'EUR_BTC_EX_High', 'Transactions_Volume', 5)
#print(check.head(5))
#check.overall_data_display
#df['cluster'] = df.apply(lambda row: assign_to_cluster(centroids_var, 'EUR_BTC_EX_High', 'Transactions_Volume', dataframe_var), axis=1)
#print(df.head(5))
#print(aaa)
#print(a)
#print(a.dataframe)
#X = df[['ones', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High']].values
#print(X)
# clustering class work end

# MachinePredictModel
# whole class
# his was the main class before the refreactored page
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
from sklearn.model_selection import ParameterGrid, GridSearchCV
from Regression import *
from ArrangeData import *
from DecisionTrees import *
from NeuralNetwork import *
from RegressionCombined import *

"""
steps
1. set all vars and load data/dbs/etc
# the needed external data should be pulled from its own pre class
# and fed here then it uses the arrange data class
2. check needed columns that have to be added changed
3. run regressions based on given 
"""

# 188, start there
class MachinePredictModel:

	# columns_all should have target name in it, both columns all and target
	# should just be the names of columns, with all being an array
	# target is target column to change
	def __init__(self, dataframe, columns_all, random_state, training_percent, kfold_number, target_col_name, **kwargs):
		self.dataframe =  dataframe
		# columns all should contain all features that will be used plus the one target
		self.columns_all = columns_all
		self.random_state = random_state
		self.training_percent = training_percent
		self.kfold_number = kfold_number
		self.target_col_name = target_col_name
		self.date_unix = date_unix = kwargs.get('date_unix', None)
		self.time_interval_check = kwargs.get('time_interval_check', None)
		self.normalize_columns_array = kwargs.get('normalize_columns_array', None)
		self.time_period_returns_dict = kwargs.get('time_period_returns_dict', None)
		self.cols_to_drop = kwargs.get('cols_to_drop', None)
		self.target = kwargs.get('target', None)
		self.target_change_bin_dict = kwargs.get('target_change_bin_dict', None)
		#self.col_to_make_target = kwargs.get('col_to_make_target', None)
		self.target_amount = kwargs.get('target_amount', None)
		self.set_multi_class = kwargs.get('set_multi_class', None)
		self.convert_unix_to_human_date = kwargs.get('convert_unix_to_human_date', None)
		self.kfold_dict = kwargs.get('kfold_dict', None)

	# this method is an interal class method to clean up date
	# what still needs to be added
	# 1. way to change needed columns data types such as turn all numerical
	# 2. set be able to set multi class vars like time of day - evebing, night etc
	# 3. drop columns if certain percent data is missing
	def _set_up_data_for_prob_predict(self, **kwargs):
		# **kawrgs vars below
		# initiate the data class
		model_dataframe = ArrangeData(self.dataframe)
		#print(time_interval_check, date_unix)
		# check if date_unix = none
		# if not none creates timedate
		if self.date_unix != None:
			model_dataframe.format_unix_date(self.date_unix)
		# this takes in an array, column name of date is first, then 1 to 
		# make new rows of the units separted by y,m,d,h,m,s,ms
		# array must have 2 variables 
		if self.convert_unix_to_human_date is not None:
			model_dataframe.convert_unix_to_human_date(self.convert_unix_to_human_date)	
		# this will eventually take in a dictionary  but first
		# the arrange data resample_date needs to be refactored for version .2 change	
		if self.time_interval_check == 1:
			model_dataframe.resample_date(self.target, 'month_highs_avg', 'M', 'mean')
			model_dataframe.resample_date(self.target, 'week_highs_avg', 'W', 'mean')
			model_dataframe.resample_date(self.target, 'day_highs_avg', 'D', 'mean')
		# normalize the given columns, with a new name which is always orginal 
		# column name + normalized
		if self.normalize_columns_array is not None:
			model_dataframe.normalize_new_column(self.normalize_columns_array)
		# takes in a dict, always has the same keys, column_name_old, column_name_new,
		# freq and returns new columns based on name of given time period return
		if self.time_period_returns_dict is not None:
			model_dataframe.time_period_returns_dict(self.time_period_returns_dict)
		if self.cols_to_drop is not None:
			model_dataframe.drop_columns(self.cols_to_drop)
		if self.target_change_bin_dict is not None:
			#model_dataframe.set_binary(self.col_to_make_target, self.target_col_name, self.target_amount)
			model_dataframe.set_binary_from_dict(self.target_change_bin_dict)
		if self.set_multi_class is not None:
			model_dataframe.set_multi_class_array(self.set_multi_class)
		model_dataframe.overall_data_display(8)
		return model_dataframe
		# everything above is setting up data, more still needs to be added
		# now comes the regressions on the bottom
		# there should be some type of dict model takes in with which models to run 
		# and which variables/error metrics to use etc
		# in fact the above method may become only class method
		# actually lets do that

	def _get_error_scores_with_tpr_fpr(self, y_target, predictions):
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
		dict ={}
		y = y_target
		dict['roc_auc_score'] = roc_auc_score(y, predictions)
		dict['mse'] = mean_squared_error(y, predictions)
		dict['mae'] = mean_absolute_error(y, predictions)
		dict['r2_score'] = r2_score(y, predictions)
		dict['variance'] = np.var(predictions)
		dict['tpr'] = true_positive_rate
		dict['fpr'] = false_positive_rate
		return(dict)

	def _get_error_scores(self, y_target, predictions):
		y = y_target
		dict ={}
		dict['roc_auc_score'] = roc_auc_score(y, predictions)
		dict['mse'] = mean_squared_error(y, predictions)
		dict['mae'] = mean_absolute_error(y, predictions)
		dict['r2_score'] = r2_score(y, predictions)
		dict['variance'] = np.var(predictions)
		return dict
		

	def predict_prob_model(self, **kwargs):
		df = self._set_up_data_for_prob_predict()
		print(self.columns_all)
		print(self.target_col_name)
		param_dict_logistic = kwargs.get('param_dict_logistic', None)
		param_dict_decision_tree = kwargs.get('param_dict_decision_tree', None)
		param_dict_neural_network = kwargs.get('param_dict_neural_network', None)
		# set up features and target
		df.shuffle_rows()
		x_y_vars = df.set_features_and_target1(self.columns_all, self.target_col_name) 
		features = x_y_vars[0]
		target = x_y_vars[1]
		# set up training and testing data
		vars_for_train_test = df.create_train_and_test_data_x_y_mixer(self.training_percent, features, target)
		X_train = vars_for_train_test[0]
		y_train = vars_for_train_test[1]
		X_test = vars_for_train_test[2]
		y_test = vars_for_train_test[3]
		ppm_results_dict = {}
		# 1st model test logistic regression
		regres_instance = Regression(features, target, self.random_state)
		if param_dict_logistic is None:
			print(' didnt pick up first kwarg')
			ppm_results_dict['log_regress_data'] = regres_instance.logistic_regres_with_kfold_cross_val()
		else:
			print('picked up params')
			ppm_results_dict['log_regress_data'] = regres_instance.logistic_regres_with_kfold_cross_val(param_dict_logistic=param_dict_logistic)
		#print(log_regress_data)

		#2nd model test decision tree
		decision_tree_instance = DecisionTree('place_holder')
		if param_dict_decision_tree is None:
			print('no vars picked up')
			ppm_results_dict['decision_tree_data'] = decision_tree_instance.basic_tree_with_vars(X_train, y_train, X_test, y_test)
		else:
			print('vars passed to decision tree class')
			ppm_results_dict['decision_tree_data'] = decision_tree_instance.basic_tree_with_vars(X_train, y_train, X_test, y_test, param_dict_decision_tree=param_dict_decision_tree)
		#print(decision_tree_data)

		# 3rd model nueral network
		if param_dict_neural_network is None:
			ppm_results_dict['nnl_instance'] = NNet3(learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4)
		else:
			nnl_instance = NNet3(learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4, param_dict_neural_network=param_dict_neural_network)
			ppm_results_dict['nnl_data'] = nnl_instance.neural_learn_sk(X_train, y_train, X_test, y_test)
		#print(nnl_data)
		return ppm_results_dict

	def predict_prob_model_full(self, **kwargs):
		# vars
		df = self._set_up_data_for_prob_predict()
		param_dict_logistic = kwargs.get('param_dict_logistic', None)
		param_dict_decision_tree = kwargs.get('param_dict_decision_tree', None)
		param_dict_neural_network = kwargs.get('param_dict_neural_network', None)
		# set up features and target
		df.shuffle_rows()
		x_y_vars = df.set_features_and_target1(self.columns_all, self.target_col_name) 
		features = x_y_vars[0]
		target = x_y_vars[1]
		vars_for_train_test = df.create_train_and_test_data_x_y_mixer(self.training_percent, features, target)
		X_train = vars_for_train_test[0]
		y_train = vars_for_train_test[1]
		X_test = vars_for_train_test[2]
		y_test = vars_for_train_test[3]
		# start prediction instace 
		predictions_instance = RegressionCombined(features, target, self.kfold_dict, X_train, X_test, y_train, y_test)
		if param_dict_logistic is None:
			print('log vars not none')
			predictions_results = predictions_instance.regression_probs_model()
		else:
			predictions_results = predictions_instance.regression_probs_model(param_dict_logistic=param_dict_logistic, param_dict_decision_tree=param_dict_decision_tree, param_dict_neural_network=param_dict_neural_network)
		return predictions_results
		#scores = self._get_error_scores_with_tpr_fpr(target, predictions_results['predictions_logistic'])
		#scores1 = self._get_error_scores_with_tpr_fpr(target, predictions_results['predictions_logistic_kfold'])
		#return scores, scores1


	def predict_prob_model_fit_parameters(self, training_percent, kfold_number, target_col_name, **kwargs):
		df = self._set_up_data_for_prob_predict()
		param_dict_logistic_array = kwargs.get('param_dict_logistic_array', None)
		param_dict_decision_tree_array = kwargs.get('param_dict_decision_tree_array', None)
		param_dict_neural_network_array = kwargs.get('param_dict_neural_network_array', None)
		# set up features and target
		df.shuffle_rows()
		x_y_vars = df.set_features_and_target1(self.columns_all, target_col_name)
		features = x_y_vars[0]
		target = x_y_vars[1]
		# set up training and testing data
		vars_for_train_test = df.create_train_and_test_data_x_y_mixer(training_percent, features,target)
		X_train = vars_for_train_test[0]
		y_train = vars_for_train_test[1]
		X_test = vars_for_train_test[2]
		y_test = vars_for_train_test[3]
		dtree = DecisionTreeClassifier()
		reg = LogisticRegression()
		nnl = MLPClassifier()
		if param_dict_logistic_array is not None:
			clf = GridSearchCV(reg, param_dict_logistic_array)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_test)
			error_score = self._get_error_scores_with_tpr_fpr(y_test, predictions)
			error_score['best_Score'] = clf.best_score_
			error_score['best_params'] = clf.best_params_
			print(error_score)
		if param_dict_decision_tree_array is not None:
			clf = GridSearchCV(dtree, param_dict_decision_tree_array)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_test)
			error_score = self._get_error_scores_with_tpr_fpr(y_test, predictions)
			error_score['best_Score'] = clf.best_score_
			error_score['best_params'] = clf.best_params_
			print(error_score)
		if param_dict_neural_network_array is not None:
			clf = GridSearchCV(nnl, param_dict_neural_network_array)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_test)
			error_score = self._get_error_scores_with_tpr_fpr(y_test, predictions)
			error_score['best_Score'] = clf.best_score_
			error_score['best_params'] = clf.best_params_
			print(error_score)

	# columns all is an array 
	def cycle_vars(self, columns_all, training_percent, kfold_number, target_col_name):
		dict = {}
		for x in range(1, len(columns_all)+1):
			kicker = x
			start = 0
			end = start+ kicker
			cols = columns_all[start:end]
			data = self.predict_prob_model_fit_parameters(training_percent, kfold_number, target_col_name)
			x +=1
			dict[str(cols)] = dataframe
		return dict

	def cycle_vars_thru_features(self, columns_all, target_col_name):
		pass
		#max = len(columns_all)-



"""
# cycle vars example
	def cycle_vars_simple_lin_regress(self, columns, target):
		results_array = []
		dict = {}
		for x in range(1, len(columns)+1):
			kicker = x
			start = 0
			end = start + kicker
			cols = columns[start:end]
			instance = ArrangeData(df)
			add = instance.simple_lin_regres(columns, target)
			results_array.append(add)
			x += 1
			dict[str(cols)] = add
		return dict
"""


"""
# info for btc_data
file_location_btc = '/home/mike/Documents/coding_all/machine_predict/btc_play_data.csv'
file_location_loans = '/home/mike/Documents/coding_all/machine_predict/cleaned_loans_2007.csv'
df = pd.read_csv(file_location_btc)
columns = ['EUR_BTC_EX_High', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
columns_all = ['target_new', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
# these columns may or may not be created but target needs to be in col list
target = 'USD_BTC_EX_High'
normalize_columns_array = ['Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
random_state = 1
# method vars
#target_amount =.05
#target_col_name = 'target_new'
#col_to_make_target = 'week_highs_avg_change'
create_target_dict = {'column_name_old':['week_highs_avg_change','3day_highs_avg_change'], 'column_name_new':['target_new', '3day_highs_avg_change_bin_value'], 'value':[.05, .01]}
columns_to_drop = []
training_percent =.08
kfold_number = 10
target_col = create_target_dict['column_name_new'][0]
#**kwargs
kwarg_dict = {'time_interval_check':1, 'date_unix':'date_unix'}
time_interval_check = 1
date_unix = 'date_unix'
time_period_returns_dict = {'column_name_old':['week_highs_avg', 'day_highs_avg'], 'column_name_new':['week_highs_avg_change', '3day_highs_avg_change'], 'freq':[1,3]}
logistic_regression_params = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':'balanced', 'random_state':random_state, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}

# target_amount=target_amount, target_col_name=target_col_name, col_to_make_target=col_to_make_target,
# error
predict = MachinePredictModel(df, columns_all, random_state,  target=target, time_interval_check=1, date_unix='date_unix', normalize_columns_array=normalize_columns_array, time_period_returns_dict=time_period_returns_dict, target_change_bin_dict=create_target_dict)
df_rdy = predict._set_up_data_for_prob_predict()
print(type(df_rdy))
df_rdy.overall_data_display(10)
predict.predict_prob_model(training_percent, kfold_number, target_col, param_dict=logistic_regression_params)
# btc end 
"""

"""
#info for loans
lend_tree_loan_data = '/home/mike/Documents/coding_all/machine_predict/cleaned_loans_2007.csv'
df_loans = pd.read_csv(lend_tree_loan_data)
columns_all_loans = ['loan_amnt', 'int_rate', 'installment', 'emp_length', 'annual_inc',
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
target_loan = 'loan_status' 
target_loan = 'loan_status'
random_state_loan = 1
training_percent_loan = .08
kfold_number_loan = 10 
logistic_regression_params_loan = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':'balanced', 'random_state':random_state_loan, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
# max depht and min samples leaf can clash 
decision_tree_params_loan = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':random_state_loan, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':'balanced', 'presort':False}
#decision_tree_params_loan = ['test']
nnl_params_loan = {'hidden_layer_sizes':(100, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':random_state_loan}
#param_dict_neural_network=nnl_params_loan
loan_predict = MachinePredictModel(df_loans, columns_all_loans, random_state_loan)
loan_predict._set_up_data_for_prob_predict()
loan_predict.predict_prob_model(training_percent_loan, kfold_number_loan, target_loan, param_dict_logistic=logistic_regression_params_loan, param_dict_decision_tree=decision_tree_params_loan,param_dict_neural_network=nnl_params_loan)
#loan_predict.predict_prob_model(training_percent_loan, kfold_number_loan, target_loan, param_dict_logistic=logistic_regression_params_loan)
"""

# info for bikes
file_location = '/home/mike/Documents/coding_all/machine_predict/hour.csv'
df_bike = pd.read_csv(file_location)
columns_to_drop_bike = ['casual', 'registered', 'dtedat']
columns_all_bike = ['instant', 'season', 'yr', 'mnth', 'hr_new', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt_binary']
create_target_dict_bike = {'column_name_old':['cnt'], 'column_name_new':['cnt_binary'], 'value':[10]}
target_bike = 'cnt_binary'
set_multi_class_bike = ['hr', 6, 12, 18, 24 , 'hr_new']
random_state_bike = 1
training_percent_bike = .08
kfold_number_bike = 10 
logistic_regression_params_bike = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':'balanced', 'random_state':random_state_bike, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
# max depht and min samples leaf can clash 
decision_tree_params_bike = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':random_state_bike, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':'balanced', 'presort':False}
#decision_tree_params_loan = ['test']
nnl_params_bike = {'hidden_layer_sizes':(100, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'random_state':random_state_bike}
kfold_dict = {'n_splits':10, 'random_state':random_state_bike, 'shuffle':False}
# bike model....
bike_predict = MachinePredictModel(df_bike, columns_all_bike, random_state_bike, training_percent_bike, kfold_number_bike, target_bike, cols_to_drop=columns_to_drop_bike,set_multi_class=set_multi_class_bike, target_change_bin_dict=create_target_dict_bike, kfold_dict=kfold_dict)
bike_predict._set_up_data_for_prob_predict()
#results = bike_predict.predict_prob_model(param_dict_logistic=logistic_regression_params_bike, param_dict_decision_tree=decision_tree_params_bike,param_dict_neural_network=nnl_params_bike)
# bike model for optimizing 
# range of values in dict form for parameters
decision_tree_array_vars = { 'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features': [None, 'auto', 'sqrt', 'log2'], 'max_depth':[2,10], 'min_samples_split':[3,50,100], 'min_samples_leaf':[1,3,5], 'class_weight':[None, 'balanced'], 'random_state':[random_state_bike]}
logistic_regression_array_vars = {'penalty':['l1','l2'], 'tol':[0.0001, .001, .01], 'C':[.02,1.0,2], 'fit_intercept':[True], 'intercept_scaling':[.1,1,2], 'class_weight':[None, 'balanced'], 'solver':['liblinear'], 'max_iter':[10,100,200], 'n_jobs':[1], 'random_state':[random_state_bike]}
neural_net_array_vars = {'hidden_layer_sizes':[(100, ),(50, )], 'activation':['relu', 'logistic', 'tanh', 'identity'], 'solver':['adam', 'lbfgs', 'sgd'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001], 'power_t':[0.5], 'max_iter':[50], 'shuffle':[True, False], 'tol':[0.0001], 'verbose':[False], 'warm_start':[False, True], 'momentum':[.1,0.9], 'nesterovs_momentum':[True], 'early_stopping':[True], 'validation_fraction':[0.1], 'beta_1':[0.9], 'beta_2':[0.999], 'epsilon':[1e-08], 'random_state':[random_state_bike]}
# optimize model 
#bike_predict.predict_prob_model_fit_parameters(training_percent_bike, kfold_number_bike, target_bike, param_dict_logistic_array=logistic_regression_array_vars, param_dict_decision_tree_array=decision_tree_array_vars, param_dict_neural_network_array=neural_net_array_vars)
#bike_predict.predict_prob_model_fit_parameters(training_percent_bike, kfold_number_bike, target_bike, param_dict_decision_tree_array=decision_tree_array_vars)
#results2 = bike_predict.predict_prob_model_full(param_dict_logistic=logistic_regression_params_bike, param_dict_decision_tree=decision_tree_params_bike, param_dict_neural_network=nnl_params_bike)
#print(results2)
results3 = bike_predict.regression_probs_model_paramter_fit(param_dict_logistic_array=logistic_regression_array_vars, param_dict_decision_tree_array=decision_tree_array_vars)
print(results3)
#columns_all_features_bike = 
#results2 = bike_predict.cycle_vars(columns_all_features_bike, training_percent_bike, kfold_number_bike, target_bike)

"""
#thoughts 
1. everything above is for classifers
2. could models be mixed and match for different decisions, such as decision tree to predict when right
and nnl to weight when wrong. 
3. need to set up to iterate over multi time periods on data and cycle thru vars 
"""

"""
how this should be
1. put in various vars on top
2. pick models to run
3. pick tpr and fpr ranges, with error compared against some
t value and return if statistically sifnifcant or not
4. print out data, which model, which params, which vars, and score (error, tpr, fpr)
5. run that on new data not seen (time period ahead)
"""

"""
whats next.....
in this order
1. refactor so everything is taken in at start of class
2. refactor all regressions to out fit, and out put regressions
3. have the error scores be part of this MachinePredictModel class and return everything as dict
4. set up method to iterate over various varaiables
5. check if chose error matetric is stat significant
6. if stat significant return return scores if they hit a certain range
"""


# MultiClassClassification.py
# whole class

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

# class takes in a x and y variable, same ones used for the neural network
# will prob change that
# this returns plots of auc score for each class
def multi_class_classification(dataframe_X, dataframe_y):
	X = dataframe_X
	y = dataframe_y

	#binarize the output 
	y = label_binarize(y, classes=[1,2,3,4,5])
	n_classes = y.shape[1]

	# add noisey features to make the problem harder
	random_state = np.random.RandomState(3)
	n_samples, n_features = X.shape
	X =  np.c_[X, random_state.randn(n_samples, 200 * n_features)]

	# traina and test data
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =.5, random_state=0)

	# predict classes against each other
	classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
	y_score = classifier.fit(X_train, y_train).decision_function(X_test)

	#compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#compute micro-average ROC curve and ROC area
	fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	# plot of ROC cruve for specified class
	# in relation to fpr, tpr
	plt.figure()
	lw = 2
	plt.plot(fpr[2], tpr[2], color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

	# plot ROC curves for the multi class problem
	# first aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

	# interploate all ROC curves at these points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# average it and compute AUC
	mean_tpr /= n_classes
	fpr['macro'] = all_fpr
	tpr['macro'] = mean_tpr
	roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

	#plot all ROC curves
	plt.figure()
	plt.plot(fpr["micro"], tpr["micro"],
	         label='micro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["micro"]),
	         color='deeppink', linestyle=':', linewidth=4)

	plt.plot(fpr["macro"], tpr["macro"],
	         label='macro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["macro"]),
	         color='navy', linestyle=':', linewidth=4)

	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(n_classes), colors):
	    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
	             label='ROC curve of class {0} (area = {1:0.2f})'
	             ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Some extension of Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	plt.show()

# testing mutli class classifcation
multi_X = a.dataframe[['ones', 'Transactions_Volume_normalized', 'Number_of_Transactions_normalized', 'LTC_BTC_EX_High_normalized', 'EUR_BTC_EX_High_normalized']].values
multi_y = a.dataframe.target_5_class.values
multi_class_classification(multi_X, multi_y)
# testing mutli class classifcation end

# neural network whole class
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

#Reression,py
# whole class
import pandas as pd
import numpy as np
import datetime
import time
from itertools import cycle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc,log_loss, precision_score
import matplotlib.pyplot as plt
import operator
# read this of above http://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
from sklearn.model_selection import train_test_split
from scipy import interp
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cross_validation import cross_val_predict, KFold, StratifiedKFold
"""
notes
1. lasso regression - https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/
2. logistic regression function with l1 penatly
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
log = LogisticRegression(penalty='l1', solver='liblinear')
log.fit(X, y)
"""

# perform regressions in this class
# much work to be done in this class
class Regression:

	def __init__(self, features, target, random_state):
		self.features = features
		self.target = target
		self.random_state = random_state

	# performs a simple linear regression 
	# can take in multiple columns
	# returns a dict with mse, mae, r2, variance 
	def simple_lin_regres(self, columns, target):
		df = self.dataframe
		cols = columns
		features = df[cols]
		target_var = df[target]
		x = features
		y = target_var
		reg = LinearRegression()
		reg.fit(x, y)
		predictions = reg.predict(x)
		dict ={}
		dict['mse'] = mean_squared_error(y, predictions)
		dict['mae'] = mean_absolute_error(y, predictions)
		dict['r2_score'] = r2_score(y, predictions)
		dict['variance'] = np.var(predictions)
		return(dict)

	# does a linear regression but with k folds test and training variance
	# kfold testing, dont know enought yet if they will be part of testing
	# methods or will be its own method like the simle train and test
	# and just keep spitting our the test variables
	# returns a dict with mse, mae, r2, variance 
	def kfold_test_simple_lin_regres(self, columns, target, fold, random_state):
		df = self.dataframe
		features = df[columns]
		target = df[target]
		variance_values = []
		mse_values = []
		ame_values =[]
		r2_score_values = []
		dict ={}
		n = len(df)
		# kfold instance
		kf = KFold(n, fold, shuffle=True, random_state = random_state)
		#iterate over the k fold
		for train_index, test_index in kf:
			#trainging and test sets
			# Make predictions on training set.
			x_train, x_test = features.iloc[train_index], features.iloc[test_index]
			y_train, y_test = target.iloc[train_index], target.iloc[test_index]
			lr = LinearRegression()
			lr.fit(x_train, y_train)
			predictions = lr.predict(x_test)
			# Compute MSE and Variance.
			mse = mean_squared_error(y_test, predictions)
			variance = np.var(predictions)
			mae = mean_absolute_error(y_test, predictions)
			r2_scores = r2_score(y_test, predictions)
			#append to array
			variance_values.append(variance)
			mse_values.append(mse)
			ame_values.append(mae)
			r2_score_values.append(r2_scores)
		dict['avg_mse'] = np.mean(mse_values)
		dict['avg_ame'] = np.mean(ame_values)
		dict['r2_score_values'] = np.mean(r2_score_values)
		dict['ave_var'] = np.mean(variance_values)
		return(dict)

	# returns dict of the simple_lin_regress metrics but
	# shows values for each combo of variables
	# this concept will be important later on for optimizing vars
	def cycle_vars_simple_lin_regress(self, columns, target):
		results_array = []
		dict = {}
		for x in range(1, len(columns)+1):
			kicker = x
			start = 0
			end = start + kicker
			cols = columns[start:end]
			instance = ArrangeData(df)
			add = instance.simple_lin_regres(columns, target)
			results_array.append(add)
			x += 1
			dict[str(cols)] = add
		return dict


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

	# run a logistic regression with kfold cross val predict
	# class_weight is how to weight the logisitc reg
	def logistic_regres_with_kfold_cross_val(self, **kwargs):
		#df = self.dataframe
		#cols = columns
		#features = df[cols]
		#target_var = df[target]
		#print(type(self.features))
		#print(type(self.target))
		param_dict = kwargs.get('param_dict_logistic', None)
		#param_dict = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':None, 'random_state':None, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
		print(param_dict)
		if param_dict is None:
			print('used default params for logistic regression')
			param_dict = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':None, 'random_state':None, 'solver':'liblinear', 'max_iter':100, 'multi_class':'ovr', 'verbose':0, 'warm_start':False, 'n_jobs':1}
		else:
			print('used user params for logistic regression')
			#param_dict= param_dict['param_dict_logistic']
		#print(param_dict)
		#print(len(param_dict))
		#print(param_dict['penalty'])
		#print(type(param_dict['penalty']))
		reg = LogisticRegression(penalty=param_dict['penalty'], dual=param_dict['dual'], tol=param_dict['tol'], C=param_dict['C'], fit_intercept=param_dict['fit_intercept'], intercept_scaling=param_dict['intercept_scaling'], class_weight=param_dict['class_weight'], random_state=param_dict['random_state'], solver=param_dict['solver'], max_iter=param_dict['max_iter'], multi_class=param_dict['multi_class'], verbose=param_dict['verbose'], warm_start=param_dict['warm_start'], n_jobs=param_dict['n_jobs'])
		#reg = LogisticRegression(random_state=self.random_state, class_weight='balanced')
		print(self.features.shape[0])
		kf =KFold(self.features.shape[0], random_state=self.random_state)
		#kf1 = StratifiedKFold(self.target, self.features.shape[0], random_state=param_dict['random_state'])
		#kf2 = StratifiedKFold(3)
		reg.fit(self.features, self.target)
		predictions = cross_val_predict(reg, self.features, self.target, cv=kf)
		tpr_fpr_rates = self._create_false_pos_and_false_neg(predictions, self.target)
		dict ={}
		y = self.target
		dict['roc_auc_score'] = roc_auc_score(y, predictions)
		dict['mse'] = mean_squared_error(y, predictions)
		dict['mae'] = mean_absolute_error(y, predictions)
		dict['r2_score'] = r2_score(y, predictions)
		dict['variance'] = np.var(predictions)
		dict['tpr'] = tpr_fpr_rates[0]
		dict['fpr'] = tpr_fpr_rates[1]
		return(dict)

	def test(self):
		df = self.dataframe
		print(type(df))


#DecisionTree
# whole class

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
#import '/home/mike/Documents/coding_all/machine_predict/ArrangeData.py'  


class DecisionTree:

	# questions for this class
	# 1. the meaning of the consistent .05 probs
	# 2. where to load the x/y vars (class v method)
	# 3. where to initiate the clf classes
	# 4. can the long hand way handle large data
	# 5. should normalized data be used on deicions trees - in this
	# case it did not seem to help
	# try these methods on other data to find error

	def __init__(self, holder_var):
		self.holder_var = holder_var
		#self.random_state = random_state

	# this needs to be fixed as random state was taken out of calss var
	def basic_tree(self, X_train, y_train, X_test, y_test):
		clf = DecisionTreeClassifier(random_state=self.random_state)
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		error = roc_auc_score(y_test, predictions)
		return error

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

	def basic_tree_with_vars(self, X_train, y_train, X_test, y_test, **kwargs):
		param_dict = kwargs.get('param_dict_decision_tree', None)
		if param_dict is None:
			print('used default params for decision tree')
			param_dict = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':None, 'max_leaf_nodes':None, 'min_impurity_split':1e-07, 'class_weight':None, 'presort':False}
		else:
			print('used user params for decision tree')
		print(param_dict)
		clf = DecisionTreeClassifier(criterion=param_dict['criterion'], splitter=param_dict['splitter'], max_depth=param_dict['max_depth'], min_samples_split=param_dict['min_samples_split'], min_samples_leaf=param_dict['min_samples_leaf'], min_weight_fraction_leaf=param_dict['min_weight_fraction_leaf'], max_features=param_dict['max_features'], random_state=param_dict['random_state'], max_leaf_nodes=param_dict['max_leaf_nodes'], min_impurity_split=param_dict['min_impurity_split'], class_weight=param_dict['class_weight'], presort=param_dict['presort'])
		#clf = DecisionTreeClassifier(criterion=param_dict['criterion'], splitter=param_dict['splitter'] )
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		y = y_test
		tpr_fpr_rates = self._create_false_pos_and_false_neg(predictions, y)
		dict = {}
		dict['roc_auc_score'] = roc_auc_score(y, predictions)
		dict['mse'] = mean_squared_error(y, predictions)
		dict['mae'] = mean_absolute_error(y, predictions)
		dict['r2_score'] = r2_score(y, predictions)
		dict['variance'] = np.var(predictions)
		dict['tpr'] = tpr_fpr_rates[0]
		dict['fpr'] = tpr_fpr_rates[1]
		return dict

	# this needs to be fixed as random state was taken out of calss var
	def random_forest_with_vars(self, X_train, y_train, X_test, y_test, \
		min_samples_leaf, n_estimators):
		clf = RandomForestClassifier(random_state=self.random_state, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		auc_error = roc_auc_score(y_test, predictions)
		mse_error = mean_squared_error(y_test, predictions)
		return auc_error, mse_error

"""
new_file_location_csv1 = '/home/mike/Documents/coding_all/data_sets_machine_predict/btc_play_data.csv'
df = pd.read_csv(new_file_location_csv1)
lend_tree_loan_data = '/home/mike/Documents/coding_all/data_sets_machine_predict/cleaned_laons_2007.csv'
df_loans = pd.read_csv(lend_tree_loan_data)

columns = ['Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']
target = 'USD_BTC_EX_High'
fold = 10
random_state = 3


# start decision tree here
# X normalized
X = a.dataframe[['ones', 'Transactions_Volume_normalized', 'Number_of_Transactions_normalized', 'LTC_BTC_EX_High_normalized', 'EUR_BTC_EX_High_normalized']].values
# X not normalized
#X = a.dataframe[['ones', 'Transactions_Volume', 'Number_of_Transactions', 'LTC_BTC_EX_High', 'EUR_BTC_EX_High']].values
y = a.dataframe.target_new.values
test_vars = test_project.create_train_and_test_data_x_y_mixer(.07, X, y)
X_train1 = test_vars[0]
y_train1 = test_vars[1]
X_test1 = test_vars[2]
y_test1 = test_vars[3]
detree = DecisionTree(random_state)
basic_tree = detree.basic_tree(X_train1, y_train1, X_test1, y_test1)
print(basic_tree)
min_var_tree = detree.basic_tree_with_vars(X_train1, y_train1, X_test1, y_test1, min_samples_split=13)
print(min_var_tree)
random_forest = detree.random_forest_with_vars(X_train1, y_train1, X_test1, y_test1, n_estimators=150, min_samples_leaf=2)
print(random_forest)



# end decision tree here 
"""

# cyber_production_start.py
# whole class
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

file_location = '/home/mike/Documents/coding_all/cyber_cur/btc_play_data.csv'
df = pd.read_csv(new_file_location)

class ArrangeData:

	# goal of class
	# display some basic data (no graphs)
	# arrange/format/edit dataframe to perform on
	# return type should always be df
	# it takes in a df and spits one out

	# the regression in this class needs to be refactored out 
	# idea to deal with large data, maybe have some methods that
	# can read into a sql db, select parameters and return
	# a temp csv to parse, aaron has an example of this

	def __init__(self, dataframe):
		self.dataframe = dataframe

	# display column names, shape, and x number of rows you want
	# needs a better display message 
	def overall_data_display(self, headnum):
		df = self.dataframe
		column_list = list(df.columns.values)
		df_total = df.shape
		column_total = df.columns.shape[0]
		print(column_list, df_total, column_total)
		# make this hve a better use friendly output message
		print(df.head(headnum))

	# print all values in a coulmn
	def print_column_all(self, column_name):
		df = self.dataframe
		x = df.loc[:,column_name]
		print(x)

	# print all columns down to number of rows specified
	def print_headers(self, count):
		df = df = self.dataframe
		print(df.head(count))

	# convert a column to a new datetime column
	# dates need to be in datetime for pandas functionality
	def convert_to_datetime(self, column_name_to_convert, new_column_name):
		df = self.dataframe
		df[new_column_name] = pd.to_datetime(df[column_name])
		return df

	# this is not working
	# goal is to be able to shift a column as some variables we want to
	# create will need this (i.e. rolling averages)
	def shift_data(self, column_name, periods):
		x = df.loc[:,column_name]
		x.shift(periods=periods)
		return df

	# drop row from df
	def drop_rows(self, number):
		df = self.dataframe
		df = df.drop(df.index[:number], inplace=True)
		return df

	# shuffle rows
	# this is needed to set up data sometimes for testing
	def shuffle_rows(self):
		df = self.dataframe
		shuffled_rows = np.random.permutation(df.index)
		df = df.loc[shuffled_rows,:]
		return df

	# returns a new column with the rolling average for the column 
	# given and the period/frequency of the window
	def rolling_average(self, column_name, periods):
		df = self.dataframe
		x = df[column_name]
		y = df['shifted'] = x.shift(-1)
		df['MA' + str(periods)] = y.rolling(window =periods).mean()
		return df

	# returns a new column with the rolling std for the column
	# given and period/ frequency given
	def rolling_std(self, column_name, periods):
		df = self.dataframe
		x = df[column_name]
		y = df['shifted'] = x.shift(-1)
		df['STD' + str(periods)] = y.rolling(window =periods).std()
		return df

	# http://blog.mathandpencil.com/group-by-datetimes-in-pandas
	# above link is for grouping by weeks, months etc from unix and datatimes
	# https://stackoverflow.com/questions/26646191/pandas-groupby-month-and-year 
	# above link also has rly good info on this 
	# this is not working
	def group_by_week(self, column_name):
		df = self.dataframe
		#df[column_name] = pd.to_datetime(df[column_name])
		"""
		#print(type(df[column_name]))
		df.groupby(pd.TimeGrouper('M'))
		return df
		x = pd.to_datetime(df[column_name])
		per = x.dt.to_period('M')
		g = df.groupby(per)
		df['month_groups'] = g.sum()
		return df
		"""
		#df['date_minus_time'] = df["_id"].apply( lambda df : 
		#datetime.datetime(year=df.year, month=df.month, day=df.day))	
		#df.set_index(df["date_minus_time"],inplace=True)
		df['new_date'] = df[column_name].apply( lambda df: datetime.datetime(year=df.year, month=df.month, day=df.day))
		df.set_index(df['new_date'], inplace=True)
		return df

	# takes in a column and creates a new column from that in binary for
	# 1 or 0, yes or no, for given target.
	# example all values above value=5 return 1 else 0
	def set_binary(self, column_name_old, column_name_new, value):
		df = self.dataframe
		y = float(value)
		df[column_name_new] = np.where(df[column_name_old] >= y, 1, 0)
		return df


	# takes in a column name, 4 target values and returns a 5 class
	# scale 1-5 
	# some thoughts on this....
	# 1. can do this be dine for any number of classes?
	# 2. have it output 1-5, strings or both
	def set_mutli_class(self, column_use, value_low, value_low_mid, value_high_mid, value_high):
		df = self.dataframe
		x = column_use
		df['target_5_class'] = 0
		mask = df[x] < value_low
		mask2 = (df[x] < value_low_mid) & (df[x] > value_low )
		mask3 = (df[x] < value_high_mid) & (df[x] > value_low_mid)
		mask4 = (df[x] > value_high_mid) & (df[x] < value_high)
		mask5 = df[x] > value_high
		df.loc[mask, 'target_5_class'] = 1
		df.loc[mask2, 'target_5_class'] = 2
		df.loc[mask3, 'target_5_class'] = 3
		df.loc[mask4, 'target_5_class'] = 4
		df.loc[mask5, 'target_5_class'] = 5
		#df.loc[mask, 'target_5_class'] = 'less than '+ str(value_low)
		#df.loc[mask2, 'target_5_class'] = 'between ' +str(value_low) + ' ' + str(value_low_mid)
		#df.loc[mask3, 'target_5_class'] = 'between ' +str(value_low_mid) + ' ' + str(value_high_mid)
		#df.loc[mask4, 'target_5_class'] = 'between ' +str(value_high_mid) + ' ' + str(value_high)
		#df.loc[mask5, 'target_5_class'] = 'greater than ' + str(value_high)
		return df

	# creates a columns of all ones
	# this is needed for certain models like neural networks
	def set_ones(self):
		df = self.dataframe
		df['ones'] = np.ones(df.shape[0])
		return df

	# creates dummy variables
	# have option to append new column, also can drop old one and both
	# can be used for something like multi class classification
	def dummy_variables(self, column_name, prefix, append=1, drop=0):
		df = self.dataframe
		dummy_var = pd.get_dummies(df[column_name], prefix=prefix)
		if append == 1 and drop != 1:
			df = pd.concat([df, dummy_var], axis=1)
			return df
		elif append == 1 and drop == 1:
			df = pd.concat([df, dummy_var], axis=1)
			df = df.drop(column_name, axis=1)
			return df
		else: 
			return dummy_var
	
	# normalize numerical data columns
	# good for mahcine learning and/or
	# different data sets have very large ranges
	# this can take in an array of columns
	def normalize_new_column(self, columns_array):
		df = self.dataframe
		#result = df.copy()
		for feature_name in columns_array:
			max_value = df[feature_name].max()
			min_value = df[feature_name].min()
			df[str(feature_name)+'_normalized'] = (df[feature_name] - min_value) / (max_value - min_value)
		return df

	# this works, needs to be refactored and ran on weekly and monthly returns
	# right now this is formating unix_dates
	# and returning some time period data
	# this needs to be refractored into two different methods
	# 1. to format unix_date
	# 2. to return columns of different time period returns
	def format_unix_date(self, column_name):
		df = self.dataframe
		#df = df.set_index([column_name])
		#df.index = pd.to_datetime(df.index, unit='s')
		x= df[column_name]
		df['Datetime'] = pd.to_datetime(x, unit='s')
		#df['Datetime'] = df['Datetime'].resample('M', how='sum')
		#GB=df.groupby([(df.index.year),(df.index.month)]).sum()
		#print(GB)
		#print(df.head(5))
		# x.unixtime = pd.to_datetime(x.unixtime, unit='s')
		w = df['Datetime']
		y =df['date_unix']
		z= df['Date']
		#w.index = pd.to_datetime(w.index, unit='s')
		#print(type(w[0]) == pd.tslib.Timestamp)
		#print(type(y[0]) == pd.tslib.Timestamp)
		#print(type(z[0]) == pd.tslib.Timestamp)
		#df['year'] = df['Datetime'].apply(lambda x: x.split('-')[1])
		#print(df.head(5))
		df.index=df['Datetime']
		df['month_highs_avg'] = df['USD_BTC_EX_High'].resample('M', how='mean')
		df['week_highs_avg'] = df['USD_BTC_EX_High'].resample('W', how='mean')
		df['day_highs_avg'] = df['USD_BTC_EX_High'].resample('D', how='mean')
		df['three_day_highs_avg'] = df['USD_BTC_EX_High'].resample('D', how='mean')
		#df['month'] = df.groupby(pd.TimeGrouper(freq='M')).sum()
		#df['month'] = pd.groupby(df,by=[df.index.month])
		#g.sum()
		#df.resample('M', how='sum')
		#print(df.head(20))
		#print(g)
		return df

	# the number has looks a certain number of days back, need to group like above
	# takes in a column and returns a new one of returns based on time period
	# this may interplay with some of the later stuff in the above method
	def time_period_returns(self, column_name_old, column_name_new, freq=1):
		df = self.dataframe
		prices = df[column_name_old]
		print(type(prices))
		df[column_name_new] = prices.pct_change(freq)
		#daily_return = prices.pct_change(1)
		#weekly_return = prices.pct_change(7)
		#monthly_return = prices.pct_change(30)
		#print(check, daily_return, weekly_return, monthly_return)
		#print(df.head(30))
		return df

	# creates a train and test set from a given X and Y
	# some thoughts on this
	# 1. will prob end in testing class
	# 2. dont know yet if this should take in x,y or take in a df
	# right now this returns 4 variables to use for testing
	def create_train_and_test_data_x_y_mixer(self, percent, X, y):
		df = self.dataframe
		highest_train_row = int(df.shape[0]*(percent))
		X_train = X[:highest_train_row]
		y_train = y[:highest_train_row]
		X_test = X[highest_train_row:]
		y_test = y[highest_train_row:]
		return X_train, y_train, X_test, y_test

	# performs a simple linear regression 
	# can take in multiple columns
	# returns a dict with mse, mae, r2, variance 
	def simple_lin_regres(self, columns, target):
		df = self.dataframe
		cols = columns
		features = df[cols]
		target_var = df[target]
		x = features
		y = target_var
		reg = LinearRegression()
		reg.fit(x, y)
		predictions = reg.predict(x)
		dict ={}
		dict['mse'] = mean_squared_error(y, predictions)
		dict['mae'] = mean_absolute_error(y, predictions)
		dict['r2_score'] = r2_score(y, predictions)
		dict['variance'] = np.var(predictions)
		return(dict)

	# does a linear regression but with k folds test and training variance
	# kfold testing, dont know enought yet if they will be part of testing
	# methods or will be its own method like the simle train and test
	# and just keep spitting our the test variables
	# returns a dict with mse, mae, r2, variance 
	def kfold_test_simple_lin_regres(self, columns, target, fold, random_state):
		df = self.dataframe
		features = df[columns]
		target = df[target]
		variance_values = []
		mse_values = []
		ame_values =[]
		r2_score_values = []
		dict ={}
		n = len(df)
		# kfold instance
		kf = KFold(n, fold, shuffle=True, random_state = random_state)
		#iterate over the k fold
		for train_index, test_index in kf:
			#trainging and test sets
			# Make predictions on training set.
			x_train, x_test = features.iloc[train_index], features.iloc[test_index]
			y_train, y_test = target.iloc[train_index], target.iloc[test_index]
			lr = LinearRegression()
			lr.fit(x_train, y_train)
			predictions = lr.predict(x_test)
			# Compute MSE and Variance.
			mse = mean_squared_error(y_test, predictions)
			variance = np.var(predictions)
			mae = mean_absolute_error(y_test, predictions)
			r2_scores = r2_score(y_test, predictions)
			#append to array
			variance_values.append(variance)
			mse_values.append(mse)
			ame_values.append(mae)
			r2_score_values.append(r2_scores)
		dict['avg_mse'] = np.mean(mse_values)
		dict['avg_ame'] = np.mean(ame_values)
		dict['r2_score_values'] = np.mean(r2_score_values)
		dict['ave_var'] = np.mean(variance_values)
		return(dict)

	# returns dict of the simple_lin_regress metrics but
	# shows values for each combo of variables
	# this concept will be important later on for optimizing vars
	def cycle_vars_simple_lin_regress(self, columns, target):
		results_array = []
		dict = {}
		for x in range(1, len(columns)+1):
			kicker = x
			start = 0
			end = start + kicker
			cols = columns[start:end]
			instance = ArrangeData(df)
			add = instance.simple_lin_regres(columns, target)
			results_array.append(add)
			x += 1
			dict[str(cols)] = add
		return dict

# this will be the clustering class, aka unsupervised machine learning
# idea of this class is two show how rows of data are related
# all the methods will return values, but for some the values are wrong
class Clustering:

	def __init__(self, dataframe, var1, var2, column_name, cluster_num=5):
		self.cluster_num = cluster_num
		self.dataframe = dataframe
		self.var1 = var1
		self.var2 = var2
		self.column_name = column_name

	# needs to be renamed
	def clustering_two_var_scatter(self):
		df = self.dataframe
		var1= self.var1
		var2= self.var2
		num_clusters = self.cluster_num
		random_initial_points = np.random.choice(df.index, size=num_clusters)
		centroids = df.loc[random_initial_points]
		# plot of all data and the random ones highlighted a diff color, in this case red
		# this below needs to bne turned into a graph method
		"""
		plt.scatter(df[var1], df[var2], c='yellow')
		plt.scatter(centroids[var1], centroids[var2], c='red')
		plt.title("Centroids")
		plt.xlabel(var1, fontsize=13)
		plt.ylabel(var2, fontsize=13)
		plt.show()
		"""
		return centroids

	def recalculate_centroids_dict(self, column_name):
		df = self.dataframe
		new_centroids_dict = dict()
		for cluster_id in range(0, self.cluster_num):
			values_in_cluster = df[df[column_name] == cluster_id]
			#calculate mean of new centroid
			new_centroid = [np.average(values_in_cluster[self.var1]), np.average(values_in_cluster[self.var2])]
			new_centroids_dict[cluster_id] = new_centroid
		return new_centroids_dict

	# from that centroid, create a dict with the id as the key and the two metrics as coordinates
	def centroids_to_dict(self):
		centroids = self.clustering_two_var_scatter()
		dictionary = dict()
		# iterating counter we use to generate a cluster_id
		counter = 0
		# iterate a pandas data frame row-wise using .iterrows()
		for index, row in centroids.iterrows():
			coordinates = [row[self.var1], row[self.var2]]
			dictionary[counter] = coordinates
			counter += 1

		return dictionary

    # using prior two methods, making a new column that assigns each row to a cluster
    # eclidean_distance giving me a dimension
    # this equations only gives one point, need to find a way
    # thru method to iterate down whole row
	def assign_to_cluster(self):
	    lowest_distance = -1
	    closest_distance = -1
	    df = self.dataframe
	    #centroids = self.clustering_two_var_scatter()
	    #closest_cluster_array =[]
	    #centroids_dict = self.centroids_to_dict()
	    centroids_dict = self.centroids_to_dict()
	    print(centroids_dict)	    
	    # centroids_dict is results from centroids_to_dict method
	    # key is counter, vaulue is coordinate
	    for cluster_id, centroid in centroids_dict.items():
	        #a = df[df[var1 == var1]]
	        #b = df[df[var2 == var2]]
	        df_row = [df[self.var1][0], df[self.var2][0]]
	        #df_row = [a,b]
	        #print(df)
	        #print(df[var1])
	        #print(len(df_row))
	        #print(len(centroid))
	        #print(type(df_row))
	        #print(type(centroid))
	        #print(df_row)
	        #print(centroid)
	        
	        # calculate distance is ethod to give distance of two coordinates
	        # right now we are caluclating each players distance from the centroid 
	        euclidean_distance = euclidean_distances(centroid, df_row)
	        #euclidean_distance = self._calculate_distance(centroid, df_row)
	        # once we have distance we want to return which centroid is closest
	        if lowest_distance == -1:
	            lowest_distance = euclidean_distance
	            closest_cluster = cluster_id
	            #print(closest_cluster)
	            #closest_cluster_array.append(closest_cluster)
	            #return closest_cluster
	            #df['cluster'] = closest_cluster 
	        elif euclidean_distance < lowest_distance:
	            lowest_distance = euclidean_distance
	            closest_cluster = cluster_id
	            #return clostest cluster
	            #closest_cluster_array.append(closest_cluster)
	            #print(closest_cluster)
	            #df['cluster'] = closest_cluster
	    return closest_cluster

	def assign_to_cluster_recalc(self):
	    lowest_distance = -1
	    closest_distance = -1
	    df = self.dataframe
	    #centroids = self.clustering_two_var_scatter()
	    #closest_cluster_array =[]
	    #centroids_dict = self.centroids_to_dict()
	    centroids_dict = self.recalculate_centroids_dict(self.column_name)    
	    print(centroids_dict)
	    # centroids_dict is results from centroids_to_dict method
	    # key is counter, vaulue is coordinate
	    for cluster_id, centroid in centroids_dict.items():
	        #a = df[df[var1 == var1]]
	        #b = df[df[var2 == var2]]
	        df_row = [df[self.var1][0], df[self.var2][0]]
	        #df_row = [a,b]
	        #print(df)
	        #print(df[var1])
	        #print(len(df_row))
	        #print(len(centroid))
	        #print(type(df_row))
	        #print(type(centroid))
	        #print(df_row)
	        #print(centroid)
	        
	        # calculate distance is ethod to give distance of two coordinates
	        # right now we are caluclating each players distance from the centroid 
	        euclidean_distance = euclidean_distances(centroid, df_row)
	        #euclidean_distance = self._calculate_distance(centroid, df_row)
	        # once we have distance we want to return which centroid is closest
	        if lowest_distance == -1:
	            lowest_distance = euclidean_distance
	            closest_cluster = cluster_id
	            #print(closest_cluster)
	            #closest_cluster_array.append(closest_cluster)
	            #return closest_cluster
	            #df['cluster'] = closest_cluster 
	        elif euclidean_distance < lowest_distance:
	            lowest_distance = euclidean_distance
	            closest_cluster = cluster_id
	            #return clostest cluster
	            #closest_cluster_array.append(closest_cluster)
	            #print(closest_cluster)
	            #df['cluster'] = closest_cluster
	    return closest_cluster

	# the second recalc is not working
	# the first df is just the random points clustered, still
	# needs to be better refined after that hance the new_dict_recalc
	# method 
	def apply_cluster_to_new_column(self):
		#first_dict = self.centroids_to_dict()
		df =self.dataframe
		column_name = self.column_name
		df[column_name] = df.apply(lambda x: self.assign_to_cluster(), axis=1)
		#second_dict = self.recalculate_centroids_dict(column_name)
		#df[column_name] = df.apply(lambda x: self.assign_to_cluster_recalc(), axis=1)
		return df

	def apply_cluster_to_new_column_recalc(self):
		#first_dict = self.centroids_to_dict()
		df =self.dataframe
		column_name = self.column_name
		df = self.apply_cluster_to_new_column()
		#second_dict = self.recalculate_centroids_dict(column_name)
		df[column_name] = df.apply(lambda x: self.assign_to_cluster_recalc(), axis=1)
		return df


	def visualize_clusters(self, dataframe, var1, var2):
		colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
		column_name = self.column_name
		df = dataframe
		num_clusters = self.cluster_num
		print(num_clusters)
		print(type(num_clusters))
		for x in range(num_clusters):
			clustered_df = df[df[column_name] == x]
			plt.scatter(clustered_df[var1], clustered_df[var2],c=colors[x-1])
			plt.xlabel(var1, fontsize=12)
			plt.ylabel(var2, fontsize=12)
		plt.show()
	    #df['cluster'] = df.apply(lambda row: assign_to_cluster(row), axis=1)
	    #return df

# neural network class
# this will be one of the biggest things we relay on
# still much to do here
# the prediction method in this class and the one right below it
# are spitting out different answers then the manual way
# still working on way
class NNet3:

    def __init__(self, learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4):
        self.learning_rate = learning_rate
        self.maxepochs = int(maxepochs)
        self.convergence_thres = 1e-5
        self.hidden_layer = int(hidden_layer)

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

# class takes in a x and y variable, same ones used for the neural network
# will prob change that
# this returns plots of auc score for each class
def multi_class_classification(dataframe_X, dataframe_y):
	X = dataframe_X
	y = dataframe_y

	#binarize the output 
	y = label_binarize(y, classes=[1,2,3,4,5])
	n_classes = y.shape[1]

	# add noisey features to make the problem harder
	random_state = np.random.RandomState(3)
	n_samples, n_features = X.shape
	X =  np.c_[X, random_state.randn(n_samples, 200 * n_features)]

	# traina and test data
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =.5, random_state=0)

	# predict classes against each other
	classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
	y_score = classifier.fit(X_train, y_train).decision_function(X_test)

	#compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#compute micro-average ROC curve and ROC area
	fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	# plot of ROC cruve for specified class
	# in relation to fpr, tpr
	plt.figure()
	lw = 2
	plt.plot(fpr[2], tpr[2], color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

	# plot ROC curves for the multi class problem
	# first aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

	# interploate all ROC curves at these points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# average it and compute AUC
	mean_tpr /= n_classes
	fpr['macro'] = all_fpr
	tpr['macro'] = mean_tpr
	roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

	#plot all ROC curves
	plt.figure()
	plt.plot(fpr["micro"], tpr["micro"],
	         label='micro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["micro"]),
	         color='deeppink', linestyle=':', linewidth=4)

	plt.plot(fpr["macro"], tpr["macro"],
	         label='macro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["macro"]),
	         color='navy', linestyle=':', linewidth=4)

	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(n_classes), colors):
	    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
	             label='ROC curve of class {0} (area = {1:0.2f})'
	             ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Some extension of Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	plt.show()

# this gets all the coin data from poloniex
# what we need to do is store the information recived in a db
# time it to the unix timestamp and have it run a certain time basis
# i think best would be a databsed for hourly and
# one for bi-day
import urllib
import requests
import json
import time

print("hi)")

command_list = ['returnTicker']
command = command_list[0]
#print(command)
response = requests.get('https://poloniex.com/public?command=' + command)
print(response.headers)
data = response.json()
ETH_GNT = data['ETH_GNT']
BTC_ETH = data['BTC_ETH']
#print(BTC_ETH)
print(data)
