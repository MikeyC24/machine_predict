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
