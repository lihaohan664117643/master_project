# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:04:17 2019

@author: ASUS
"""

import numpy as np
from sklearn.model_selection import train_test_split
from myxgb import myxgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
np.random.seed(0)
#X = np.linspace(-2*np.pi,2*np.pi,4000,endpoint=True)
X = np.linspace(0,2*np.pi,1000,endpoint=True)
Y_0 = np.exp(0.6*X)*np.sin(X)
Y_1=np.sin(X)
#Y_0=Y_0+np.random.normal(1,0.5,1000)
#Y_1=Y_1+np.random.normal(1,0.5,1000)
fig=plt.figure(figsize=(12, 7)) 
ax1 = fig.add_subplot(111)
ax1.set_title('sine functiom')
ax1.scatter(X,Y_0,color='blue',label='np.exp(2*X)*np.sin(X)')
ax1.scatter(X,Y_1,color='red',label='NP.SIN(X)')
ax1.set_xlabel('predictive values')
ax1.set_ylabel('Real Target Values')
plt.legend(loc='best')
plt.show()
train_X, test_X, train_Y, test_Y = train_test_split(X, Y_0, test_size=0.3, random_state=1)
linbst1 = myxgb(loss_function="reg:squarederror",#
                n_estimators=1,#num_trees=3 and lbda=0.的时候出现 can't do matrix inverse
                learning_rate=0.01,
                min_samples_leaf=10,
                max_samples_linear_model=10000,
                max_depth=2,
                subsample=1,
                colsample_bytree=1,
                lbda=0.02,
                gamma=0,
                random_state=1,
                prune=False,
                verbose=1)
#linbst1.fit(train_X[:,:], train_Y[:])


linbst1.fit(train_X.reshape(-1,1), train_Y,test_X.reshape(-1,1),test_Y)
y_pred=linbst1.predict(test_X.reshape(-1,1))
print('MSE为：',mean_squared_error(test_Y,y_pred))
fig=plt.figure(figsize=(12, 7)) 
ax1 = fig.add_subplot(111)
ax1.set_title('sine functiom')
#ax1.scatter(train_X,train_Y,color='blue',label='training')
ax1.scatter(test_X,y_pred,color='red',label='test_pred')
ax1.scatter(test_X,test_Y,color='green',label='test_true')
ax1.set_xlabel('predictive values')
ax1.set_ylabel('Real Target Values')
plt.legend(loc='best')
plt.show()

param = {'booster': 'gbtree', # gbtree, gblinear
         'eta': 0.05, # step size shrinkage
         'objective': 'reg:squarederror' # binary:logistic, reg:linear
         }
num_round = 150 # the number of round to do boosting, the number of trees
dtrain = xgb.DMatrix(train_X.reshape(-1,1), label=train_Y.reshape(-1,1))
bst1 = xgb.train(param, dtrain, num_round)
dtest = xgb.DMatrix(test_X.reshape(-1,1))
bst1=bst1.predict(dtest)
print('MSE为：',mean_squared_error(test_Y,bst1))
fig=plt.figure(figsize=(12, 7)) 
ax1 = fig.add_subplot(111)
ax1.set_title('sine functiom')
#ax1.scatter(train_X,train_Y,color='blue',label='training')
ax1.scatter(test_X,bst1,color='red',label='test_pred')
ax1.scatter(test_X,test_Y,color='green',label='test_true')
ax1.set_xlabel('predictive values')
ax1.set_ylabel('Real Target Values')
plt.legend(loc='best')
plt.show()

#xgboost seem to fit better with the sine function without any noise, does that means xgboost can't 
#recongnize the noise better than gbdt-pl