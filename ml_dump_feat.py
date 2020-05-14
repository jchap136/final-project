#!/usr/bin/env python3

"""
Usage: ./ml_dump_feat.py total_data_0.1.csv finalized_model_x.sav

use a model made from ml_data_4.py
"""

# 1. import
import sys
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import make_scorer, balanced_accuracy_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_validate, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor
import sklearn.ensemble as ses
from scipy.spatial.distance import euclidean, minkowski
from imblearn.metrics import sensitivity_specificity_support, sensitivity_score
import imblearn.ensemble as ime
import joblib


# put file into DataFrame
f = open(sys.argv[1])
data = pd.read_csv(f, sep='\t')
df = pd.DataFrame(data)
df.drop(df.columns[[0,1,2,3,8,9,17,18,21,22]], axis = 1, inplace=True) # drop name columns
print(df)
print(df.isna().sum())

# either drop or impute missing values
dfd = df.dropna() # drop missing values

# put labels into DataFrames for either regression or classification
dfd_reg = dfd['KD_vs_NT_log2FC'] # labels for regression
dfd_class = dfd['UPF1-substrate'] # labels for classification
dfd.drop(dfd.columns[[0,6,7,17,18]], axis=1, inplace=True) # drop target_id column, replicates, and label columns
print(dfd.isna().sum())
head = list(dfd.columns.values)
print(head)

# put values into numpy arrays
x = dfd.values # returns a numpy array
print(x.shape)
y = dfd_class.values # returns a numpy array
print(y.shape)

# scale data
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(x)

# stratified train/test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=245)
train_index, test_index = tuple(sss.split(X, y))[0]
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# get saved joblib model
filename = sys.argv[2]
loaded_model = joblib.load(filename)

feat_imp = pd.DataFrame(loaded_model.feature_importances_, columns=['scores'])
feat_imp['features'] = head
feat_imp.set_index('features', inplace=True)
feat_imp['log2_scores'] = np.log2(feat_imp['scores'])
feat_imp.sort_values('scores', ascending=False, inplace=True)
print(feat_imp)

fig, ax = plt.subplots()
ax.bar(feat_imp.index.values, feat_imp['scores'], color='blue', alpha=0.75)
ax.set_ylabel('feature importance')
ax.set_xlabel('features')
ax.set_xticklabels(feat_imp.index.values, rotation=45, ha='right')

plt.show()
plt.close()






