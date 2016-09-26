# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:07:02 2016

@author: loredp
"""

import numpy.linalg as npl
import numpy as np
import math

 
def phi(X,M,f):
    n = len(X)
    phi_m = np.zeros((n,M+1))
    for i in range(n):
        for j in range(M+1):
            phi_m[i,j] = f(X[i],j)
    return phi_m
    
def poli(x,n):
    return x**n
    
def cos(x,n):
    return np.cos(math.pi*x*n)
    
def ml_weight(X,Y,M,basis,l=0):
    '''
    Calculates the weights for linear regression
    Parameters:
    X
    Y
    M
    basis
    l: lambda for regularization. If omitted is assumed to be 0
    '''
    m = phi(X,M,basis)  
    m_t = np.matrix.transpose(m)
    p1 = npl.inv(l*np.eye(M+1) + np.dot(m_t,m))
    p2 = np.dot(m_t,Y)
    return np.dot(p1,p2)