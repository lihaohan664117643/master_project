# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 23:40:22 2019

@author: ASUS
"""

import sys
import numpy as np
from tree_node import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from scipy.special import expit

class myxgb:
    split_pos_matrix=None
    def __init__(self, loss_function="reg:squarederror", n_estimators=5,
                 min_samples_split=3, min_samples_leaf=2, max_depth=6,
                 max_samples_linear_model=sys.maxsize,
                 subsample=1.0,colsample_bytree=1.0,
                 learning_rate=0.3, min_split_loss=0.0, gamma=0.0, lbda=0.0,
                 prune=True,tree_method='hist',
                 random_state=None,
                 verbose=0, nthread=1):

        self.loss_function = loss_function
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_samples_linear_model = max_samples_linear_model
        self.subsample = subsample
        self.colsample_bytree=colsample_bytree
        self.learning_rate = learning_rate
        self.min_split_loss = min_split_loss
        self.lbda = lbda
        self.gamma = gamma
        self.prune = prune
        self.tree_method='hist'
        self.random_state = random_state
        self.verbose = verbose
        self.nthread = nthread
    def squareloss(self, y, y_hat):
        """Return the squared loss wo/ penalty / regularization.
        """
        n=len(y)
        return (np.sum(np.square(y_hat-y)))

    def dsquareloss(self, X, y, y_hat):
        """Return the first-order derivative of the squared loss
        """
        n=len(y)
        return (2*(y_hat-y))
    
    def ddsquareloss(self, X, y, y_hat):
        """Return the second-order derivative of the squared loss
        """
        
        n = len(y)
        return (2*np.ones(n, dtype=float))
    
    def logisticloss(self, y, y_hat):
        """Return the logisitc loss wo/ penalty / regularization.
        """
        
        return np.sum(y*np.log(1.+np.exp(-y_hat)) + (1.-y)*np.log(1.+np.exp(y_hat)))
        
    def dlogisticloss(self, X, y, y_hat):
        """Return the first-order derivative of the logistic loss
        """
        
        return -( (y-1.)*np.exp(y_hat)+y)/(np.exp(y_hat)+1.)
        
    def ddlogisticloss(self, X, y, y_hat):
        """Return the second-order derivative of the logistic loss
        """
        
        return np.exp(y_hat)/np.square(np.exp(y_hat)+1.)
    
    def multisoftmaxloss(self,y,y_hat):
        n,k=y.shape
        fm=np.exp(y_hat)#(n,k)
        norm=np.sum(np.exp(y_hat),axis=1)#(n,)
        p=fm/np.tile(norm.reshape(-1,1),k)#(n,k)
        
        return -np.sum((np.sum(np.log(p)*y,axis=1)))
    
    def dmultisoftmaxloss(self,X,y,y_hat):
        
        n,k=y.shape
        fm=np.exp(y_hat)#(n,k)
        norm=np.sum(np.exp(y_hat),axis=1)#(n,)
        p=fm/np.tile(norm.reshape(-1,1),k)#(n,k)
        return -(y-p)#(n,k)
    
    def ddmultisoftmaxloss(self,X,y,y_hat):
        n,k=y.shape
        fm=np.exp(y_hat)#(n,k)
        norm=np.sum(np.exp(y_hat),axis=1)#(n,)
        p=fm/np.tile(norm.reshape(-1,1),k)#(n,k)
        return p-np.power(p,2)#(n,k)
    
    
    
    def regularization(self): 
        """Return the penalty for all trees built so far.
        """

        reg = 0.
        for tree in self.trees:
            reg += tree.regularization(gamma=self.gamma, lbda=self.lbda)

        return reg
    
    def objective(self,X, y, y_hat=None): #计算linxgboost所有的objective function
        if y_hat is None:
            y_hat = self._predict(X,y)
        return self.loss_func(y,y_hat)+self.regularization() #loss_func计算出来是一个常数
    
    def create_split_pos_matrix(self,X_original,tree_method='hist',max_bin=255):#v2 换了一种设置histgram的方法
        n, d = X_original.shape
        split_pos_matrix=[]
        if tree_method=='hist':
            for f in range(0,d):
                bin_num=np.unique(X_original[:,f]).shape[0]
                if bin_num>max_bin:
                    boundry=np.histogram(X_original[:,f],255)[1]
                else:
                    boundry=np.histogram(X_original[:,f],bin_num)[1]
                split_pos_matrix.append(boundry)
                
        return split_pos_matrix
    
    def build_tree(self, tree, X, g, h,new_X=None,grow_policy='depthwise'):
        assert tree.left == None, "the node must be a leaf!"
        n, d = X.shape
        if tree.parent is None:#根节点
            X1=np.concatenate((X,np.ones(shape=(n,1),dtype=np.float64)),axis=1)
            tree.set_weight(X,g,h,self.lbda,new_X=X1)#注意根节点的feature0_value值不是新加上去的feature的值
            fea_new=np.dot(X1[:,:-1],tree.w[:-1])
            new_X[:,0]=fea_new
        else:#如果不是根节点的话传进来的nex_X一定不是none
            try:
                tree.set_weight(X, g, h, self.lbda,new_X=new_X)
                new_X[:,0]=tree.feature0_value
            except:
                    print( "in tree building: something went wrong!" )
                    raise
        if grow_policy=='depthwise':
            if tree.depth >= self.max_depth:
                return tree
            if n < self.min_samples_split:
                return tree

            tree.find_best_split(X, g, h, self.lbda, self.gamma, self.max_samples_linear_model, self.min_samples_leaf,self.tree_method,myxgb.split_pos_matrix,new_X)#传进去的new_X是一个n*3的矩阵
            if tree.best_split_feature == -1: # no split because of the constraints
                if self.verbose>2:
                    print( "node could not be split" )
                return tree
            left_child, right_child = tree.add()
            c = ( X[:,tree.split_feature] < tree.split_value+0.05 )
            #如果使用下列代码可能会在递归的过程中占用很多内存因为前一个递归总是握着后一个递归的x_tmp地址
            #X_tmp=np.compress(c,X,axis=0)
            #g_tmp=np.compress(c,g,axis=0)
            #h_tmp=np.compress(c,h,axis=0)
            #new_X_tmp=np.compress(c,new_X,axis=0)
            if self.verbose>1:
                
                print( "creating left child with {:d} instances".format(np.sum(c)) )
            try:
                self.build_tree(left_child, np.compress(c,X,axis=0), np.compress(c,g,axis=0), np.compress(c,h,axis=0),np.compress(c,new_X,axis=0),grow_policy='depthwise')#找到最佳的叶子节点之后，继续构建当前节点的左节点
            except RuntimeError:
                print( "maximum recursion depth exceeded: Tree depth is {}".format(tree.depth) )
                tree.left = None
                tree.right = None
                return tree
                raise
            c = ( X[:,tree.split_feature] > tree.split_value-0.05 )
#            X_tmp=np.compress(c,X,axis=0)
#            g_tmp=np.compress(c,g,axis=0)
#            h_tmp=np.compress(c,h,axis=0)
#            new_X_tmp=np.compress(c,new_X,axis=0)
            if self.verbose>1:
                
                print( "creating right child with {:d} instances".format(np.sum(c)) )
            try:
                self.build_tree(right_child, np.compress(c,X,axis=0),np.compress(c,g,axis=0),np.compress(c,h,axis=0),np.compress(c,new_X,axis=0),grow_policy='depthwise')#找到最佳的叶子节点之后，继续构建当前节点的右节点
            except RuntimeError:
                print( "maximum recursion depth exceeded: Tree depth is {}".format(tree.depth) )
                tree.left = None
                tree.right = None
                return tree
                raise
                
            return tree
 #,X_evl,y_evl       
    def fit(self, X, y,X_evl,y_evl):
        X=X.astype(np.float64)
        X_evl=X_evl.astype(np.float64)
       # y=y.astype(np.float64)
        #y_evl.astype(np.float64)
       
        if self.loss_function == "reg:squarederror":
            self.loss_func = self.squareloss
            self.dloss_func = self.dsquareloss
            self.ddloss_func = self.ddsquareloss
        elif self.loss_function == "binary:logistic":
            self.loss_func = self.logisticloss
            self.dloss_func = self.dlogisticloss
            self.ddloss_func = self.ddlogisticloss
        elif self.loss_function=="multi:softmax":
            self.loss_func = self.multisoftmaxloss
            self.dloss_func = self.dmultisoftmaxloss
            self.ddloss_func = self.ddmultisoftmaxloss
        else:
            raise ValueError("unknown error function")
            
        if y.ndim != 1:
            print( "fit() is expecting a 1D array!" )
            y = y.ravel()   
        assert X.shape[0] == y.shape[0], 'x and y is not in the same length'
        self.trees = []
        self.tree_objs = []
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        #self.tree_objs.append( self.objective(X,y) )
        if self.loss_function=="multi:softmax":

            for t in range(0,self.n_estimators):
                n,d = X.shape
                batch_size = int(np.rint(self.subsample*n))
                indices = np.random.choice(n, batch_size, replace=False)
                np.random.seed(t)
                col_batch_size=int(np.rint(self.colsample_bytree*d))
                col_indices=np.random.choice(d,col_batch_size,replace=False)
                X_tmp=np.take(X,indices,axis=0)
                X_tmp=np.take(X_tmp,col_indices,axis=1)
                y_tmp=np.take(y,indices,axis=0)
                myxgb.split_pos_matrix=self.create_split_pos_matrix(X_tmp,tree_method='hist',max_bin=255)
                #tree_root=node(verbose=self.verbose)
                #one hot code for y
                k=np.unique(y_tmp).shape[0]
                y_tmp=np.eye(batch_size,k)[y_tmp]
                y_hat = self._predict(np.take(X,indices,axis=0),y_tmp)
                y_pred=self.predict(X_evl)
                
                g = self.dloss_func(X_tmp,y_tmp,y_hat)
                h = self.ddloss_func(X_tmp,y_tmp,y_hat)
                #tree_root.col_indices=col_indices
                ktree_obj=0
                ktree=[]
                print( " we are now building tree {}, total obj={}".format(t+1,np.sum(self.tree_objs)) )
                for ki in range(0,k):
                    tree_root=node(verbose=self.verbose)
                    tree_root.col_indices=col_indices
                    new_X=np.zeros((batch_size,3))
                    new_X[:,2]=1
                    tree = self.build_tree( tree_root, X_tmp, g[:,ki], h[:,ki],new_X=new_X,grow_policy='depthwise' )#根节点已经传进去一个n*3的new_X
                    tree.recalculate(X_tmp,g[:,ki],h[:,ki],self.lbda)
                    ktree.append(tree)
                    tree_obj = tree.objective(self.gamma)
                    
                    if self.prune:
                        num_pruning = self.prune_tree_type_2(tree)
                        if num_pruning > 0:
                            tree_obj = tree.objective(self.gamma)
                            if self.verbose > 0:
                                print("{} nodes pruned, ".format(num_pruning))
                                
                            tree.recalculate(X_tmp,g,h,self.lbda)
                    ktree_obj+=tree_obj
                    
                    
                    
                self.trees.append(ktree)  #有n_estimator个ktree， 每个ktree又有K个tree  
                self.tree_objs.append(ktree_obj)
                
            return self
        
        for t in range(0,self.n_estimators):
            n,d = X.shape
            batch_size = int(np.rint(self.subsample*n))
            indices = np.random.choice(n, batch_size, replace=False)
            
            np.random.seed(t)
            col_batch_size=int(np.rint(self.colsample_bytree*d))
            col_indices=np.random.choice(d,col_batch_size,replace=False)
            X_tmp=np.take(X,indices,axis=0)
            X_tmp=np.take(X_tmp,col_indices,axis=1)
            y_tmp=np.take(y,indices,axis=0)
            myxgb.split_pos_matrix=self.create_split_pos_matrix(X_tmp,tree_method='hist',max_bin=255)
            tree_root=node(verbose=self.verbose)
            
            y_hat = self._predict(np.take(X,indices,axis=0),y_tmp)
            y_pred=self.predict(X_evl)
            if self.loss_function == "reg:squarederror":
                
                    print('at iteration{} evluation mse is {}'.format(t+1,mean_squared_error(y_pred,y_evl)))
                    print('at iteration{} training mse is {}'.format(t+1,mean_squared_error(y_hat,y_tmp)))
            else:
                    print('at iteration{} acc is {}'.format(t+1,accuracy_score(y_evl,expit(y_pred)>0.5*1)))
            g = self.dloss_func(X_tmp,y_tmp,y_hat)
            h = self.ddloss_func(X_tmp,y_tmp,y_hat)
            
            tree_root.col_indices=col_indices
           #w_root,obj_root=tree_root.get_weight(None,g,h,self.lbda,new_X=X[indices,:])# since this is the root of a tree it doesn't have any parents
            #fea_new=np.dot(np.c_[X,np.ones(shape=(batch_size,1), dtype=float)],w_root)
            #new_X=fea_new.reshape(batch_size,-1)#这行代码可要可不要
            print( " we are now building tree {}, total obj={}".format(t+1,np.sum(self.tree_objs)) )
            new_X=np.zeros((batch_size,3),dtype=np.float64)
            new_X[:,2]=1
            tree = self.build_tree( tree_root, X_tmp, g, h,new_X=new_X,grow_policy='depthwise' )#根节点已经传进去一个n*3的new_X
            tree.recalculate(X_tmp,g,h,self.lbda)
            self.trees.append(tree)
            tree_obj = tree.objective(self.gamma)
            print( "This tree{} max. depth={}, num. of leaves={}, obj_{}={}, total obj={}". \
                       format(t+1,tree.max_depth(),tree.num_leaves(), t+1, tree_obj, np.sum(self.tree_objs)+tree_obj) )
            #还要加tree pruning 和清除Obg小于0的树
            
            # pruning
            if self.prune:
                num_pruning = self.prune_tree_type_2(tree)
                if num_pruning > 0:
                    tree_obj = tree.objective(self.gamma)
                    if self.verbose > 0:
                        print("{} nodes pruned, ".format(num_pruning))
                        print( "After pruning the tree{} max. depth={}, num. of leaves={}, obj_{}={}, total obj={}". \
                               format(t+1,tree.max_depth(),tree.num_leaves(), t+1, tree_obj, np.sum(self.tree_objs)+tree_obj) )
                tree.recalculate(X_tmp,g,h,self.lbda)
            self.tree_objs.append(tree_obj)
            
            # check obj是否小于0 和数是否是叶子
            if tree_obj > 0. or self.trees[-1].is_leaf():
                if self.verbose > 0:
                    print( "the objective of tree {}/{} of depth {} is positive: obj = {:.4e}". \
                            format(t+1,self.n_estimators,tree.max_depth(),tree_obj) )
                #print(self.trees[-1].is_leaf())
                del self.trees[-1]
                del self.tree_objs[-1]
                break
        return self
    
    def _predict(self, X,y_true):
        """Make predictions.

        """

        n = X.shape[0]
        if self.loss_function=='reg:squarederror':
            #y=np.mean(y_true)*np.ones(n,dtype=float)
            y = np.zeros(n, dtype=float)
        else:
            
            y = np.zeros(n, dtype=float)
        if self.loss_function=="multi:softmax":
            #k=np.unique(y_true).shape[0]
            k=y_true.shape[1]
            yk=np.zeros((n,k),dtype=float)
            for ktrees in self.trees:
                ki=0
                for tree in ktrees:
                    yk[:,ki]+=self.learning_rate*tree.predict(X[:,tree.col_indices])
                    ki+=1
            return yk
            
        else:
            
            for tree in self.trees:
            
                y += self.learning_rate*tree.predict(X[:,tree.col_indices])
        
            return y
    def predict(self, X):
        """Make predictions.
        """

        n = X.shape[0]
        y = np.zeros(n, dtype=float)
        if not self.trees:
            return y
        if self.loss_function=="multi:softmax":
            k=len(self.trees[0])
            yk=np.zeros((n,k),dtype=float)
            for ktrees in self.trees:
                ki=0
                for tree in ktrees:
                    yk[:,ki]+=self.learning_rate*tree.predict(X[:,tree.col_indices])
                    ki+=1
            return yk
        else:
            
            for t in range(len(self.trees)-1):
                tree=self.trees[t]
            
                y += self.learning_rate*tree.predict(X[:,tree.col_indices])
        
            y += self.trees[-1].predict(X[:,self.trees[-1].col_indices])
        

        
            return y
    
    def prune_tree_type_2(self, tree):
        num_pruning = 0
        if not tree.is_leaf():
            if tree.gain < 0.:
                if tree.obj < tree.objective(gamma=self.gamma): #在一颗子树中如果起根节点的obj比整颗子树的obj还要小，证明这个切分没用
                    tree.left = None
                    tree.right = None
                    num_pruning += 1
            if not tree.is_leaf():
                
                num_pruning += self.prune_tree_type_2(tree.left)
                num_pruning += self.prune_tree_type_2(tree.right)
        return num_pruning
    
    
    
    
    
