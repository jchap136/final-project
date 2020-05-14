#!/usr/bin/env python3

"""
Usage: ./ml_dump.py total_data_0.1.csv finalized_model_x.sav

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

result_list = []
result_list.append(loaded_model.score(X_test, y_test))

for i in range(x.shape[1]):
    
    dfd_c = dfd.copy()
    dfd_c.iloc[:,i] = np.nan
    dfd_c.fillna(0, inplace=True)
    
    x = dfd_c.values
    X = scaler.fit_transform(x)
    
    train_index, test_index = tuple(sss.split(X, y))[0]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    result = loaded_model.score(X_test, y_test)
    result_list.append(result)


print(result_list)
print(len(result_list))

a = pd.Series(result_list)
print(a)

plt.plot(a)
plt.xlabel('dropped feature')
plt.ylabel('score')
plt.show()
plt.close()




