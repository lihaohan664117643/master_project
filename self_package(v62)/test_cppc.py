# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 21:27:51 2019

@author: ASUS
"""

import numpy as np
import pandas as pd
from myxgb import myxgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.special import expit
from sklearn.model_selection import train_test_split

df = pd.read_excel('ccpp.xlsx')
multivariate_outliers1 = df[ (df["V"]<30) | (df["PE"]<424) ].index.tolist()
multivariate_outliers2 = df[ (df["V"]>70) & (df["V"]<73) & (df["PE"]>450) & (df["PE"]<480) ].index.tolist()
multivariate_outliers = multivariate_outliers1 + multivariate_outliers2
new_df = df.drop(df.index[multivariate_outliers]).reset_index(drop = True)
data = new_df.values
print(data.shape)
features = data[:,:-1]
features = features - np.mean(features,axis=0)
target = data[:,-1]

train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.3, random_state=1)
linbst1 = myxgb(loss_function="reg:squarederror",#
                n_estimators=200,#num_trees=3 and lbda=0.的时候出现 can't do matrix inverse
                learning_rate=0.1,
                min_samples_leaf=32,
                max_samples_linear_model=10000,
                max_depth=15,
                subsample=1,
                colsample_bytree=0.7,
                lbda=1e-11,
                gamma=1,
                random_state=1,
                prune=False,
                verbose=1)
#linbst1.fit(train_X[:,:], train_Y[:])


linbst1.fit(train_X[:,:], train_Y[:],test_X,test_Y)
y_pred=linbst1.predict(test_X)
print('MSE为：',mean_squared_error(test_Y,y_pred))


param = {'booster': 'gbtree', # gbtree, gblinear
         'eta': 0.05, # step size shrinkage
         'objective': 'reg:squarederror', # binary:logistic, reg:linear
         'max_depth':10
         }
num_round = 2000 # the number of round to do boosting, the number of trees
dtrain = xgb.DMatrix(train_X, label=train_Y.reshape(-1,1))
bst1 = xgb.train(param, dtrain, num_round)
dtest = xgb.DMatrix(test_X)
bst1=bst1.predict(dtest)
print('MSE为：',mean_squared_error(test_Y,bst1))