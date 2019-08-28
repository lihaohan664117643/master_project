# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:49:27 2019

@author: ASUS
"""

from myxgb import myxgb
from scipy.special import expit
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

breast=pd.read_csv('./breast/mysubtrain.csv',header=None)
train_X=breast.iloc[:,:-1]
train_Y=breast.iloc[:,-1]
linbst = myxgb(loss_function="binary:logistic",#
                n_estimators=30,#num_trees=3 and lbda=0.的时候出现 can't do matrix inverse
                learning_rate=0.05,
                min_samples_leaf=5,
                max_samples_linear_model=10000,
                max_depth=5,
                subsample=1,
                colsample_bytree=0.8,
                lbda=0.01,
                gamma=1,
                prune=False,
                random_state=1,
                verbose=1)
#linbst.fit(train_X.as_matrix(), train_Y.as_matrix())

test_X=pd.read_csv('./breast/mytest.csv',header=None)
test_Y=pd.read_csv('./breast/mytest_label.csv',header=None)

linbst.fit(train_X.as_matrix(), train_Y.as_matrix(),test_X.as_matrix(),test_Y.as_matrix())
results=linbst.predict(test_X.as_matrix())

print('the acc rate of the linxgboost with 1 trees is {}'.format(accuracy_score(test_Y,expit(results)>0.5*1)))