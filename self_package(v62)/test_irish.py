# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:13:41 2019

@author: ASUS
"""

import numpy as np
import pandas as pd
from myxgb import myxgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

iris=pd.read_csv('IRIS.csv')
target=iris.iloc[:,-1]
features=iris.iloc[:,:-1]
le = preprocessing.LabelEncoder()
le.fit(target)
target=le.transform(target)
train_X, test_X, train_Y, test_Y = train_test_split(features.as_matrix(), target, test_size=0.2, random_state=1,stratify=target)
linbst1 = myxgb(loss_function="multi:softmax",#
                n_estimators=50,#num_trees=3 and lbda=0.的时候出现 can't do matrix inverse
                learning_rate=0.01,
                min_samples_leaf=1  ,
                max_samples_linear_model=10000,
                max_depth=8,#4或者5的时候最佳，此时不需要设置columsubsample
                subsample=1,
                colsample_bytree=0.6,
                lbda=0.01,
                gamma=0,
                random_state=1,
                prune=False,
                verbose=1)
#linbst1.fit(train_X[:,:], train_Y[:])


linbst1.fit(train_X[:,:], train_Y[:],test_X,test_Y)
y_pred=linbst1.predict(test_X)
fm=np.exp(y_pred)#(n,k)
norm=np.sum(np.exp(y_pred),axis=1)#(n,)
p=fm/np.tile(norm.reshape(-1,1),3)#(n,k)
y_pred_label=np.argmax(p,axis=1)
#print('MSE为：',mean_squared_error(test_Y,y_pred))
accuracy_score(test_Y,y_pred_label)


param = {'booster': 'gbtree', # gbtree, gblinear
         'eta': 0.01, # step size shrinkage
         'objective': 'multi:softmax', # binary:logistic, reg:linear
         'num_class':3,
         'max_depth':3
         }
num_round = 1 # the number of round to do boosting, the number of trees
dtrain = xgb.DMatrix(train_X, label=train_Y.reshape(-1,1))
bst1 = xgb.train(param, dtrain, num_round)
dtest = xgb.DMatrix(test_X)
bst1=bst1.predict(dtest)
accuracy_score(test_Y,bst1)
#print('MSE为：',mean_square