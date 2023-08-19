# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 10:51:21 2023

@author: khushal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('eda_data.csv')

# TASKS :- 
# choose relevant columns
# get dummy data
# train test split

# multiple linear regression
# lasso regression
# random forest
# tune models gridsearchCV
# test ensembles

# choose relevant columns
df.columns
df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided',
             'job_state','same_state','age','python_yn','spark','aws','excel','job_simp','seniority','desc_len']]

# get dummy data
df_dum = pd.get_dummies(df_model)

# train test split - just google - sklearn train test split -> https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#:~:text=from%20sklearn.model_selection%20import%20train_test_split
from sklearn.model_selection import train_test_split

# X - without avg_salary so model don't know it
X = df_dum.drop('avg_salary', axis =1)
# y - array of avg salary
y = df_dum.avg_salary.values

# 0.2 means 80% in our train set and 20% in test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# multiple linear regression 
# - "statesmodel ols regression" - https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html#:~:text=import%20statsmodels.api%20as%20sm
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

# "sklearn linear regression example" - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=a%20predictor%20object.-,Examples,-%3E%3E%3E%20import%20numpy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#:~:text=from%20sklearn.model_selection%20import%20cross_val_score
# cross_val_score -> it generates and validates itself
# neg_mean_absolute_error -> how far or off than general prediction
# cross validation is set to three
np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
# with it we can sat on avg we are 20k off

# lasso regression 
from sklearn.linear_model import Lasso
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#:~:text=alphafloat%2C%20default%3D1.0
# try diff val of alpha and seeing
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

"""
# finding alpha value
alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]
"""

#lm_l = Lasso(alpha=.13)
#lm_l.fit(X_train,y_train)
#np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

'''

# random forest - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=from%20sklearn.ensemble%20import%20RandomForestRegressor
# we expect random f. to perform well here because it's kind of tree based decision process
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))

# tune models GridsearchCV - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#:~:text=*%20n_jobs.-,Examples,-%3E%3E%3E%20from%20sklearn
from sklearn.model_selection import GridSearchCV
# n_estimators - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=n_estimatorsint%2C%20default%3D100
# criterion - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=criterion%7B%E2%80%9Csquared_error%E2%80%9D%2C%20%E2%80%9Cabsolute_error,when%20using%20%E2%80%9Csquared_error%E2%80%9D.
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
# Somewhat i get error in fitting data
gs.fit(X_train,y_train)

gs.best_score_ # -14... if works
gs.best_estimator_

'''

# test ensembles, i think below code is not going to run because our data doesn't fit in 110 line.
# but we modified it so we can use this with linear regression..
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = lm.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_rf)

mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)

import pickle
pickl = {'model': lm}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]

list(X_test.iloc[1,:])
