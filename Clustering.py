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