# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:07:02 2016

@author: loredp
"""

import numpy.linalg as npl
import numpy as np
import math

def poli(x,n):
    return x**n

def phi(X,M,f=poli):
    n = len(X)
    phi_m = np.zeros((n,M+1))
    for i in range(n):
        for j in range(M+1):
            phi_m[i,j] = f(X[i],j)
    return phi_m
     
def cos(x,n):
    return np.cos(math.pi*x*n)
    
def ml_weight(X,Y,M = 1,basis = poli ,l=0):
    '''
    Calculates the weights for linear regression
    Parameters:
    X: vecotr of Nx1
    Y: target vector of Nx1
    M: dimensions of the function basis
    basis: class of the basis
    l: lambda for regularization. If omitted is assumed to be 0
    '''
    m = phi(X,M,basis)  
    m_t = np.matrix.transpose(m)
    p1 = npl.inv(l*np.eye(M+1) + np.dot(m_t,m))
    p2 = np.dot(m_t,Y)
    return np.dot(p1,p2)
    
def SSE(w,L):
    '''
    Calculate sum of square residuals.
    w: column vector resulting that represents the weights 
    L is a dictinoary with the following values:
    'X': X vector of Nx1
    'Y' = target vector of Nx1    
    OPTIONAL:
    'M': dimension of the basis, otherwsie it assumes 1
    'basis': basis por the transformation, otherwise it assumes poli
    'l': lambda for regularization, otherwise it assumes 0
    '''
    #Extract parameters
    X = L.get('X')
    Y = L.get('Y')
    M = L.get('M',1)
    l = L.get('l',0) 
    basis = L.get('basis',poli)
    y_pred = np.dot(phi(X,M,basis),w)  
    w_t = np.matrix.transpose(w)
    return np.dot(np.matrix.transpose(Y-y_pred),Y-y_pred) + l*np.dot(w_t,w)   
    
