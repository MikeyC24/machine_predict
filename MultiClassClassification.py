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