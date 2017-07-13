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
from Regression import *
from ArrangeData import *
from DecisionTrees import *

lend_tree_loan_data = '/home/mike/Documents/coding_all/data_sets_machine_predict/cleaned_loans_2007.csv'
df_loans = pd.read_csv(lend_tree_loan_data)

loans_data = ArrangeData(df_loans)
loans_data.overall_data_display(1)