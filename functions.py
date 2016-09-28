# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:07:02 2016

@author: loredp
"""

import numpy.linalg as npl
import numpy as np
import math
import matplotlib.pyplot as plt


# Some basis functions  


def central_difference(f,x,h): 
    '''Using central difference to approximate the gradient of a function f:R^n->R
    at point x.
    '''
    n = len(x)
    m = np.zeros([n,1])
    for i in range(n):
        h0 = np.asmatrix(np.zeros([n,1]))
        h0[i] += h
        m[i] = (f(x + h0) - f(x-h0))/(2*h)
    return np.matrix(m)
    
def poli(x,n):
    ''' 
    Polinomial basis
    '''
    return x**n    
     
def cos(x,n):
    return np.cos(math.pi*x*n)
    
def q2(x):
    return np.cos(x*math.pi)+np.cos(x*2*math.pi)    

# Linear regression 

def phi(X,M,f=poli):
    '''
    Map X to the basis function determined by f with dimension M.
    Input: 
    X: vector of nx1
    M: maximum order of the basis (2 for polinomial implies 1, x, x^2)
    f: basis function
    Output:
    nx(M+1) desisgn matrix (Bishop 3.16)
    '''
    n = len(X)
    phi_m = np.zeros((n,M+1))
    for i in range(n):
        for j in range(M+1):
            phi_m[i,j] = f(X[i],j)
    return phi_m
    
def ml_weight(X, Y, M = 1, l = 0, basis = poli):
    '''
    Calculates the weights for linear regression
    Parameters:
    X: vector of Nx1
    Y: target vector of Nx1
    M: dimensions of the function basis
    basis: class of the basis. Default basis set to polinomial.
    l: lambda for regularization. If omitted is assumed to be 0
    '''
    m = phi(X,M,basis)  
    m_t = np.matrix.transpose(m)
    p1 = npl.inv(l*np.eye(M+1) + np.dot(m_t,m))
    p2 = np.dot(m_t,Y)
    return np.dot(p1,p2)
        
def SSE(X,Y,w,M,l = 0, basis = poli):
    '''
    Sum of squared erros
    X: X vector of Nx1
    Y = target vector of Nx1   
    Calculate sum of square residuals.
    w: column vector resulting that represents the weights 
    OPTIONAL:
    M: dimension of the basis, otherwsie it assumes 1
    basis: basis por the transformation, otherwise it assumes poli
    l: lambda for regularization, otherwise it assumes 0
    OUTPUT:
    function of SSE that only depends on 
    '''
    def fun_SSE(w):
         y_pred = np.dot(phi(X,M,basis),w)  
         w_t = np.matrix.transpose(w)
         res = np.dot(np.matrix.transpose(Y-y_pred),Y-y_pred) + l*np.dot(w_t,w)
         return res[0]
    return fun_SSE   

def SSE_grad(X,Y,w,M = 1,l = 0, basis = poli): 
    '''
    Gradient for the sum of squared errors
    X: X vector of Nx1
    Y = target vector of Nx1   
    Calculate sum of square residuals.
    w: column vector resulting that represents the weights 
    OPTIONAL:
    M: dimension of the basis, otherwsie it assumes 1
    basis: basis por the transformation, otherwise it assumes poli
    l: lambda for regularization, otherwise it assumes 0
    OUTPUT:
    function of SSE that only depends on 
    '''
    def grad_SSE(w):       
        y_pred = np.dot(phi(X,M,basis),w)  
        #w_t = np.matrix.transpose(w)
        return -2*np.dot(np.matrix.transpose(phi(X,M,basis)),(Y-y_pred))
        
    return grad_SSE
    
#h = 1e-05*np.ones(len(w))    
    
    
def evaluate(X,Y,M = 1, w = [], l = 0, basis = poli, plot = False, f = None):
    '''Evaluates a single model for given M, l, basis and data.
    Plots the adjusted model when plot = True
    Returns the corresponding weights (closed-form solution) and SSE
    '''
    if w == []:
        w = ml_weight(X,Y,M,l=l)
        
    f_SSE = SSE(X,Y,w,M,l,basis)
    
    if plot == True:
        x_plot = np.linspace(X.min(),X.max(),100)
        y_plot = np.dot(phi(x_plot,M,basis),w)      
        plt.plot(X,Y,'o')
        plt.plot(x_plot,y_plot)
        if f != None:
            y_true= f(x_plot) 
            plt.plot(x_plot,y_true)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    return w, f_SSE(w)
    
def model_eval(X,Y,M_list,lambda_list, basis = poli):
    '''
    Given a training set X,Y and a list of M and lambda values, model_eval
    the value of the weights for each model.
    The output is a mxl matrix of lists where m is the number of M values and l is the
    number of l values. The entry i,j represents the weights for model with M[i] and 
    lambda[j]]
    '''
    aux = []
    for m in M_list:
        for ll in lambda_list:
            w = ml_weight(X,Y,m,ll,basis)
            aux.append(w)
    W = np.matrix(aux)
    return np.reshape(W,(len(M_list),len(lambda_list)))
    
def model_select(X,Y,M_list,lambda_list, W, basis = poli):
    '''
    Given a validation set X,Y and a list of M and lambda values, and its 
    corresponding weights obtained in model_eval, model_select returns the SSE
    for the adjusted model. 
    W is the matrix obtained in model_eval.
    The output is a mxl matrix. The entry i,j represents the SSE for model
    with M[i] and lambda[j]]
    '''
    aux = []
    for i in range(len(M_list)):
        for j in range(len(lambda_list)):
            d = {'X':X,'Y':Y,'M':M_list[i],'l':lambda_list[j],'basis':basis}
            w = W[i,j]
            res = SSE(w,d)
            aux.append(res)
    R = np.matrix(aux)
    return np.reshape(R,(len(M_list),len(lambda_list)))
        
def train(train_data, val_data, M_list, lambda_list, basis = poli):
    '''
    Optimizes the parameterse in the training data and later evaluates
    in the validation data for a given grid of M and lambda values.
    Recibes train_data and val_data as tuples with X,Y values
    '''
    W = model_eval(train_data[0],train_data[1],M_list,lambda_list, basis = poli)
    SSE = model_select(val_data[0],val_data[1],M_list,lambda_list, W, basis = poli)
    
    #Create a grid for helping read the results
    
    g = np.matrix(['M: ' + str(i) + "lambda: " + str(j) for i in M_list for j in lambda_list])
    grid = np.reshape(g,(len(M_list),len(lambda_list)))
    
    return W,SSE,grid
