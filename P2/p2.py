# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:29:48 2016

@author: loredp
"""

from loadFittingDataP2 import getData
import numpy.linalg as npl
import numpy as np
import matplotlib.pyplot as plt
import math

data = getData(ifPlotData=False)

X = data[0][:,None]
Y = data[1][:,None]

def ml_weight(X,Y,M,basis):
    m = phi(X,M,basis)  
    m_t = np.matrix.transpose(m)
    p1 = npl.inv(np.dot(m_t,m))
    p2 = np.dot(m_t,Y)
    return np.dot(p1,p2)
 
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

    
    
    
    
w = ml_weight(X,Y,10)    

x = np.arange(0,1,.02)

y_pred = np.zeros(len(x))
for i in range(len(theta)):
    y_pred += x**i*theta[i]
    
y_real = np.cos(x*math.pi)+np.cos(x*2*math.pi)

plt.plot(X,Y,'o')
plt.plot(x,y_pred)
plt.plot(x,y_real)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# SSE

M = 2


    
    
w = ml_weight(X,Y,M)   

def SSE(w,L):
    '''
    Calculate sum of square residuals.
    L is a tuple that contains:
    X = L[0] vector of Nxm
    Y = L[1] vector of Nx1
    M = L[2] dimension of the polynomial basis
    '''
    X = L[0]
    Y = L[1]
    M = L[2]
    y_pred = np.dot(phi(X,M),w)  
    return np.dot(Y-y_pred,np.matrix.transpose(Y-y_pred))   
    
    
res = SSE(w,(X,Y,M))
    

def SSE_grad(X,Y,w):
    phi_mat = phi(X,M)
    y_pred = np.dot(phi_mat,w)  
    return -2*np.dot(np.matrix.transpose(phi_mat),np.matrix.transpose(Y-y_pred))
    
h = 1e-05*np.ones(len(w))    




