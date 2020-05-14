#!/usr/bin/env python3

"""
Usage: ./ml_data_model_select.py total_data.csv

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
from sklearn.metrics import make_scorer, balanced_accuracy_score, accuracy_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_validate, StratifiedShuffleSplit
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

################################################################################################################################

# make scoring and kfold instance
# scoring = make_scorer(balanced_accuracy_score) # also run with this scoring method
scoring = make_scorer(f1_score)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
it = 25

################################################################################################################################

# KNN Classifier
knn = KNeighborsClassifier()
p_grid = {
          "n_neighbors": range(1,30),
          "leaf_size": range(1,50)
}
gknn = RandomizedSearchCV(knn, p_grid, n_iter=it, cv=kf, scoring=scoring)
gknn.fit(X, y)
cv_knn = gknn.cv_results_['mean_test_score']
print(cv_knn)
df_knn = pd.DataFrame(cv_knn)
df_knn.columns = ['KNN']


###############################################################################################################################

# Random Forest Classifier
rf = ses.RandomForestClassifier()
p_grid = {
     "max_depth": range(1, 25),
     "n_estimators": range(15, 75),
     "min_samples_leaf": range(1, 25),
     "min_samples_split": range(2, 25)
}
grf = RandomizedSearchCV(rf, p_grid, n_iter=it, cv=kf, scoring=scoring)
grf.fit(X, y)
cv_rf = grf.cv_results_['mean_test_score']
print(cv_rf)
df_rf = pd.DataFrame(cv_rf)
df_rf.columns = ['RandomForest']

# Balanced Random Forest Classifier
brf = ime.BalancedRandomForestClassifier()
p_grid = {
     "max_depth": range(1, 25),
     "n_estimators": range(15, 75),
     "min_samples_leaf": range(1, 25),
     "min_samples_split": range(2, 25)
}
gbrf = RandomizedSearchCV(
                          brf,
                          p_grid,
                          n_iter=it,
                          cv=kf,
                          scoring=scoring)
gbrf.fit(X, y)
cv_brf = gbrf.cv_results_['mean_test_score']
print(cv_brf)
df_brf = pd.DataFrame(cv_brf)
df_brf.columns = ['BalancedRandomForest']

###############################################################################################################################

# Gradient Boosting Classifier
gb = ses.GradientBoostingClassifier()
p_grid = {
     "max_depth": range(1, 25),
     "n_estimators": range(15, 75),
     "min_samples_leaf": range(1, 25),
     "min_samples_split": range(2, 25)
}
ggb = RandomizedSearchCV(gb, p_grid, n_iter=it, cv=kf, scoring=scoring)
ggb.fit(X, y)
cv_gb = ggb.cv_results_['mean_test_score']
print(cv_gb)
df_gb = pd.DataFrame(cv_gb)
df_gb.columns = ['GradientBoosting']

###############################################################################################################################

# Easy Ensemble Classifier: Bag of balanced boosted learners
ee = ime.EasyEnsembleClassifier()
p_grid = {"n_estimators": range(2, 50)}
gee = RandomizedSearchCV(ee, p_grid, n_iter=it, cv=kf, scoring=scoring)
gee.fit(X, y)
cv_ee = gee.cv_results_['mean_test_score']
print(cv_ee)
df_ee = pd.DataFrame(cv_ee)
df_ee.columns = ['EasyEnsemble']

###############################################################################################################################

# Random under-sampling integrating in the learning of an AdaBoost classifier
rb = ime.RUSBoostClassifier()
p_grid = {"n_estimators": range(2, 100)}#, "learning_rate": range(1,10)}
grb = RandomizedSearchCV(rb, p_grid, n_iter=it, cv=kf, scoring=scoring)
grb.fit(X, y)
cv_rb = grb.cv_results_['mean_test_score']
print(cv_rb)
df_rb = pd.DataFrame(cv_rb)
df_rb.columns = ['RUSBoost']

###############################################################################################################################

# Adaboost classifier
ab = ses.AdaBoostClassifier()
p_grid = {"n_estimators": range(2, 100)}#, "learning_rate": range(1,10)}
gab = RandomizedSearchCV(ab, p_grid, n_iter=it, cv=kf, scoring=scoring)
gab.fit(X, y)
cv_ab = gab.cv_results_['mean_test_score']
print(cv_ab)
df_ab = pd.DataFrame(cv_ab)
df_ab.columns = ['AdaBoost']

# ################################################################################################################################
#
# # # Bagging classifier with KNN base estimator: takes too long though
# # bk = ses.BaggingClassifier(KNeighborsClassifier())#, max_samples=0.5, max_features=0.5)
# # p_grid = {
# #           "base_estimator__n_neighbors": range(1,30),
# #           "base_estimator__leaf_size": range(1,50)
# # }
# # gbk = RandomizedSearchCV(bk, p_grid, n_iter=10, cv=kf, scoring=scoring)
# # gbk.fit(X, y)
# # cv_bk = gbk.cv_results_['mean_test_score']
# # print(cv_bk)

# # practice data
# df1 = pd.DataFrame([1,2,3,3,3,3,2,2,2,2,])
# df1.columns = ['a']
# df2 = pd.DataFrame([7,8,7,6,7,8,9,9,8,9])
# df2.columns = ['b']
# df_list = [df1,df2]
# labels = ['a','b','a']

df_list = [df_knn, df_rf, df_gb, df_ab, df_ee, df_brf, df_rb]
df = pd.concat(df_list, axis=1)
df_m = pd.melt(df)
df_m.columns = ['model', 'F1 score']
print(df_m)

labels = ['KNN', 'RandomForest', 'GradientBoosting', 'AdaBoost', 'EasyEnsemble', 'BalancedRandomForest', 'RUSBoost']

ax = sns.swarmplot(data=df_m, x='model', y='F1 score', order=labels)
ax.set_xticklabels(rotation=30, labels=labels)
plt.show()
plt.close()




