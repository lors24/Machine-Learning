# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 12:00:19 2016

@author: Omar
"""

import numpy as np
from loadParametersP1 import getData
import math
from loadFittingDataP1 import getData as getFittingData
import random

def gradient_descent(f,f_prime,x0,step,treshold,maxiter = 500):
    x = x0
    iter = 0
    while np.linalg.norm(f_prime(x)) > treshold & iter < maxiter:
        x = x - step*f_prime(x)
        iter += 1
    return x
    
parametersP1 = getData()
fittingData = getFittingData()
u = np.matrix.transpose(np.asmatrix(parametersP1[0]))
sigma = np.asmatrix(parametersP1[1])
A = np.asmatrix(parametersP1[2])
b = np.matrix.transpose(np.asmatrix(parametersP1[3]))

def gaussian_function(x):
    n = len(u)
    return (-1/(math.sqrt(((2*math.pi)**n))*np.linalg.det(sigma)))*math.exp(-np.matrix.transpose(x-u)*np.linalg.inv(sigma)*(x-u)/2)
    
def gaussian_function_prime(x):
    return -gaussian_function(x)*np.linalg.inv(sigma)*(x-u)
    
def quadratic_bowl(x):
    return 0.5*np.matrix.transpose(x)*A*x - np.matrix.transpose(x)*b
    
def quadratic_bowl_prime(x):
    return A*x - b
    
def finiteDif(f,h):
    def fun(x):
        a = np.asmatrix(f(x))
        n = len(a)
        p = len(x)
        m = np.zeros([n,p])
        if n > 1:
            for i in xrange(n):
                for j in xrange(p):
                    h0 = np.zeros([n,1])
                    h0[j] += np.matrix(h)/2
                    m[i][j] = (f(x + h0)[i] - f(x- h0)[i])/h

        else:
            for j in xrange(p):
                h0 = np.asmatrix(np.zeros([p,1]))
                h0[j][0] += h/2
                m[0][j] = (f(x + h0) - f(x - h0))/h
        return np.matrix.transpose(np.matrix(m))
    return fun

X = np.asmatrix(fittingData[0])
y = np.transpose(np.asmatrix(fittingData[1]))

def J(theta):
    return np.linalg.norm(X*theta - y)
    
def J_prime(theta):
    return 2*np.transpose(X)*(X*theta - y)
    
theta_check = np.linalg.inv((np.transpose(X)*X))*np.transpose(X)*y
x0 = np.matrix(np.zeros([10,1]))
gradient_descent(J,J_prime,x0, 1e-8, 1e-6)

def j(theta,x,y):
    return (np.matrix.transpose(x)*theta - y)**2
    
def j_prime(theta,x,y):
    return 2*x*(np.matrix.transpose(x)*theta - y)
    
def j_theta(theta):
    x = np.transpose(X[1])
    l = y[1]
    return j(theta,x,l)
    
def J_prime_partial(theta,A,b):
    return 2*np.transpose(A)*(A*theta - b)
    
def nu(t):
    k = 0.9
    nu0 = 1e08
    return (nu0 + t)**(-k)
    
    
    SSE_grad(X,Y,M = 1, basis = poli)
    
#J_prime = SSE_s  
#X = phi(X)    
    

    
    
def SGD2(k,criterion):
    n,m = np.shape(X)
    theta = np.matrix(np.ones([m,1]))
    t = 1
    while np.linalg.norm(J_prime(theta)/n) > criterion:
        sample = random.sample(range(n),k)
        x = X[sample,:]
        l = y[sample]
        theta = theta - nu(t)*J_prime_partial(theta,x,l)
        t += 1
    return theta