# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 18:54:39 2016

@author: loredp
"""
import functions as f
from loadFittingDataP2 import getData
import regressData as rg

#import numpy.linalg as npl
#import matplotlib.pyplot as plt
import numpy as np
#import math

data = getData(ifPlotData=True)

X = data[0][:,None]
Y = data[1][:,None]
M = 2
l = 0.5

w, res = f.eval(X,Y,M = M,l=l,plot = True)
print "The SSE is " + str(res)


## 3.2

A = rg.regressAData()
B = rg.regressBData()
v = rg.validateData()

X1 = A[0]
Y1 = A[1]
X2 = B[0]
Y2 = B[1]

M_list = [1,2,3,5,8,10]
lambda_list = [0,0.5,1]

mod_A = f.model_eval(X1,Y1,M_list,lambda_list)
mod_B = f.model_select(X2,Y2,M_list,lambda_list, mod_A)
        
g = np.matrix(['M: ' + str(i) + "lambda: " + str(j) for i in M_list for j in lambda_list])
grid = np.reshape(g,(len(M_list),len(lambda_list)))
 
weights, errors, grid = f.train(A, v, M_list, lambda_list)

w, RSS_final = eval(B[0],B[1], weights[5,1], M = 10, l = 0.5, plot = True)