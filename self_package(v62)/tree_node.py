# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 00:51:00 2019

@author: ASUS
"""

import numpy as np
import gc
class node:
    tree_method='hist'

    def __init__(self,verbose=0):
        self.parent=None
        self.left=None
        self.right=None
        self.split_value=0.0
        self.split_feature=-1
        self.feature0_value=None#存放feature0 的值
        self.col_indices=None
        self.indices=None
        self.best_split_feature=-1
        self.split_pos_matrix=None
        self.w=0
        self.depth=0
        self.gain=0.0
        self.obj=0.0
        self.verbose=verbose
        
    def is_leaf(self):
        return  self.left==None and self.right==None
    
    def max_depth(self):
        if self.is_leaf():
            return self.depth
        else:
            return max( self.left.max_depth(), self.right.max_depth() )
        
    def num_leaves(self):
        if self.is_leaf():
            return 1
        else:
            return self.left.num_leaves()+self.right.num_leaves()
        
    def add(self):
        assert self.is_leaf(), "y must add children to a leaf!"

        left_node = node(verbose=self.verbose)
        left_node.parent = self
        left_node.depth = self.depth+1
        self.left = left_node

        right_node = node(verbose=self.verbose)
        right_node.parent = self
        right_node.depth = self.depth+1
        self.right = right_node

        return self.left, self.right
    
    def delete_node(self):
        assert self.is_leaf(), "you should delete a leaf!"
        assert self.depth != 1, "you cannot delete root!"

        parent = self.parent
        parent.left = None
        parent.right = None

        return parent
    
    def objective(self, gamma=0.): #计算一棵树，一颗subtrees或者一个叶子节点的objective loss
        
        if self.is_leaf():
            return self.obj+gamma
        else:
            return self.left.objective(gamma)+self.right.objective(gamma)
        
    def regularization(self, gamma=0., lbda=0.):#计算一棵树，一颗subtrees或者一个叶子节点的objective loss
        if self.is_leaf():
            if np.isscalar(self.w):  #为什么要这个
                # in that case, lambda was set to 0 (no penalty for the bias)
                w2 = 0.
            else:
                w2 = np.dot(self.w,self.w)
            return gamma + 0.5*lbda*w2
        else:
            return self.left.regularization(gamma,lbda)+self.right.regularization(gamma,lbda)
        

                
            
        
        
    def set_weight(self, X,g, h, lbda,new_X=None):
        self.w, self.obj,self.feature0_value = self.get_weight(X, g, h, lbda,new_X,set_weights=True)#linear_model 是一个boolean
    
    def get_weight(self,X,g,h,lbda,new_X=None,set_weights=False):
        try:
            n= X.shape[0]
        except:
            print('new_X is None')
            raise
        if self.parent!=None and set_weights:#当只有在set_weight和不是根节点的时候
            #X_tilde=np.c_[new_X,X[:,self.parent.best_split_feature],np.ones(shape=(n,1),dtype=float)]
            new_X[:,1]=X[:,self.parent.best_split_feature]
            X_tilde=new_X
            d=new_X.shape[1]
        else:
            
            X_tilde = new_X
            d=new_X.shape[1]
        
        g_tilde = np.dot(X_tilde.transpose(),g)
        H_tilde = np.dot(X_tilde.transpose()*h, X_tilde)
        Lambda = lbda*np.eye(d)
        Lambda[d-1,d-1] = 0.
        C = H_tilde+Lambda
        try:
            C_inv=np.linalg.inv(C)
             #在这里算obj的时候没有加上gamma *T ，
        except:
            print('Cannot do matrix inverse')
            raise
        
        w=-np.dot(C_inv,g_tilde)
        
        obj=0.5*np.dot(g_tilde.transpose(),w)
        return w,obj,np.dot(new_X[:,:-1],w[:-1]) #w(m,) obj:scalar
    
    def find_best_split(self, X, g, h, lbda, gamma, max_samples_linear_model, min_samples_leaf,tree_method,split_pos_matrix,new_X):
        assert new_X.shape[0] == len(g), 'length of the data and g is not the same'
        assert len(h) == len(g), 'length of h and g are not the same'
        assert self.split_feature == -1, 'y have found the split featrue before'
        n, d = X.shape# n 是batchsize大小 d是原始数据集的所有特征大小
        self.gain = np.float64("-inf")
        if tree_method=='hist':
            for f in range(0,d):
                #print('now finding the feature{}'.format(f))
                for pos in split_pos_matrix[f]:
                    
                    c = ( X[:,f] < pos )
                    left_n = np.sum(c) #计算左叶子的数目
                    right_n = n-left_n#计算右叶子的数目
                    if ( left_n < min_samples_leaf ) or ( right_n < min_samples_leaf ):#如果叶子的数目小于min_samples_leaf则跳过这一split
                        continue
                    feature=X[:,f]
                    #X_for_linear=np.c_[new_X,feature]
                    new_X[:,1]=feature
                    X_for_linear=new_X
                    
                    #X_for_linear送入左叶子节点进行线性回归的matrix
                    try:
                        _ , obj_left,feature0_value = self.get_weight(np.compress(c,X,axis=0), np.compress(c,g,axis=0),np.compress(c,h,axis=0), lbda,np.compress(c,X_for_linear,axis=0)) #获取左叶子节点的objection数值
                    except:
                        if self.verbose > 1:
                            print( "exception when testing for split of feature {} at pos {}".format(f,pos) )
                        continue
                    #del X_for_linear
                    #gc.collect()
                    c = np.invert(c)
                    try:
                        _ , obj_right,feature0_value = self.get_weight(np.compress(c,X,axis=0), np.compress(c,g,axis=0),np.compress(c,h,axis=0), lbda,np.compress(c,X_for_linear,axis=0))#获取右节点的objection
                    except:
                        if self.verbose > 1:
                            print( "exception when testing for split of feature {} at pos {}".format(f,pos) )
                        continue
                    
                    
                    gain = self.obj-(obj_left+obj_right+gamma) #到时候再看下gamma是＋还是－
                    if gain > self.gain:
                        self.gain = gain
                        self.split_feature = f
                        self.split_value = pos
            
            self.best_split_feature=self.split_feature    
            if self.verbose > 1:
                if ( self.gain != np.float64("-inf") ) and ( self.gain < 0. ):
                    print( "negative gain!" )
                print( "find best split: gain={:+6.4e}, feature={:2d}, value={:+8.4f}".format(self.gain, self.split_feature, self.split_value) )
            
    def predict(self,X):
        n,d = X.shape
        X_tmp=X
        if self.is_leaf():
            fea_flag=np.zeros(d,dtype=int)
            node=self.parent
            X_leaf=np.ones(n,dtype=float).reshape(-1,1)
            while(node is not None):
                if fea_flag[node.best_split_feature]!=1:
                    X_leaf=np.concatenate((X_tmp[:,node.best_split_feature].reshape(-1,1),X_leaf),axis=1)
                    fea_flag[node.best_split_feature]=1
                node=node.parent
            if np.isscalar(self.w):
                return self.w
            else:
                return np.dot(X_leaf,self.w)
        else:
            assert self.split_feature > -1, "split feature must be > -1!"
            y = np.zeros(n, dtype=float)
            
            c = ( X[:,self.split_feature] < self.split_value )

            y[c]            = self.left.predict(X_tmp[c,:])
            y[np.invert(c)] = self.right.predict(X_tmp[np.invert(c),:])
            return y
    def recalculate(self,X,g,h,lbda):
        n,d=X.shape
       # w=np.zeros(d,dtype=float)
        #X_tmp=np.concatenate((X,np.ones(shape=(n,1),dtype=float)),axis=1)
        X_tmp=X
        if self.is_leaf():
            fea_flag=np.zeros(d,dtype=int)
            node=self.parent
            X_tilde=np.ones(n,dtype=float).reshape(-1,1)
            while(node is not None):
                if fea_flag[node.best_split_feature]!=1:
                    X_tilde=np.concatenate((X_tmp[:,node.best_split_feature].reshape(-1,1),X_tilde),axis=1)
                    fea_flag[node.best_split_feature]=1
#
                node=node.parent
            d= X_tilde.shape[1]
            g_tilde = np.dot(X_tilde.transpose(),g)
            H_tilde = np.dot(X_tilde.transpose()*h, X_tilde)
            Lambda = lbda*np.eye(d)
            Lambda[d-1,d-1] = 0.
            C = H_tilde+Lambda
            try:
                C_inv=np.linalg.inv(C)
             #在这里算obj的时候没有加上gamma *T ，
            except:
                print('Cannot do matrix inverse')
                raise
        
            self.w=-np.dot(C_inv,g_tilde)
        else:
            assert self.split_feature > -1, "split feature must be > -1!"
            c = ( X[:,self.split_feature] < self.split_value )
            self.left.recalculate(X_tmp[c,:],g[c],h[c],lbda)
            self.right.recalculate(X_tmp[np.invert(c),:],g[np.invert(c)],h[np.invert(c)],lbda)