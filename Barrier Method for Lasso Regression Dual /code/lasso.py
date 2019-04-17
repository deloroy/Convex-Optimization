#Convex Optimization -Homework 3
#Author : Yonatan Deloro

import numpy as np

def build_QP_lasso(X,Y,lambd):
    # X,Y : data, labels
    # lambd : weight of L1 regularization in lasso problem
    # returns (Q,p,A,b)
    (n,d) = X.shape
    Q = 0.5 * np.eye(n) #(n,n)
    p = Y #(n,)
    A = np.hstack([X,-X]).T #(2d,n)
    b = lambd * np.ones(2*d) #(2d,)
    return [Q,p,A,b]

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

def generate_data(n=100,d=100,sparsity_w=0.5,seed=0):
    # returns X,Y,w
    # X : (n,d) n random samples of dimension d
    # Y : (n,1) computed as Xw
    # where w : (d,1) has only int(d*(1-sparsity_w)) non-zeros random entries
    np.random.seed(seed)
    N_non_zeros_entries = int((1-sparsity_w)*d)
    X = 2*np.random.rand(n, d)-1
    w = np.zeros(d)
    w[:N_non_zeros_entries] = 2*np.random.rand(N_non_zeros_entries)-1
    Y = X.dot(w)
    return X,Y,w