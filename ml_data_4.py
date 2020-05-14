#!/usr/bin/env python3

"""
Usage: ./ml_data.py total_data.csv

check current working data table and start machine learning
1. import
2. instantiate
3. fit
4. predict
5. score
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
y_reg = dfd_reg.values # returns a numpy array
print(y_reg.shape)
y = dfd_class.values # returns a numpy array
print(y.shape)

# scale x values so they are all on the same scale (between 0 and 1) and account for outliers
#https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py

# scale data
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(x)

# make scoring and kfold instance
scoring = make_scorer(balanced_accuracy_score)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Balanced Random Forest Classifier
brf = ime.BalancedRandomForestClassifier()
p_grid = {
     "max_depth": range(1, 25),
     "n_estimators": range(10, 200),
     "min_samples_leaf": range(1, 25),
     "min_samples_split": range(2, 25)
}

# Find optimal hyperparameters
g = RandomizedSearchCV(brf, p_grid, n_iter=50, cv=kf, scoring=scoring)
g.fit(X, y)
best = g.best_params_
cv = g.cv_results_

# Balanced Random Forest Classifier with optimal hyperparameters
# can change this to brf2 = g.best_estimator_ instead of refitting
brf2 = ime.BalancedRandomForestClassifier(
                                        max_depth = best['max_depth'],
                                        n_estimators = best['n_estimators'],
                                        min_samples_leaf = best['min_samples_leaf'],
                                        min_samples_split = best['min_samples_split']
)

# make a stratified train/test split for testing final model performance
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_index, test_index = tuple(sss.split(X, y))[0]
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# train model with training data and score model
brf2.fit(X_train, y_train)
acc = balanced_accuracy_score(y_test, brf2.predict(X_test))
acc = round(acc, 4)
print(acc)

# save model with joblib
filename = 'finalized_model_{}.sav'.format(acc)
joblib.dump(brf2, filename)





