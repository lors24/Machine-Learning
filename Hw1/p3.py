# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 18:54:39 2016

@author: loredp
"""
import functions as f
from loadFittingDataP2 import getData
import regressData as rg
import numpy as np

data = getData(ifPlotData=True)

## 3.2

A = rg.regressAData()
B = rg.regressBData()
v = rg.validateData()

X_train = A[0]
Y_train = A[1]
X_test = B[0]
Y_test = B[1]
X_val = v[0]
Y_val = v[1]

M_list = [1,2,5,10]
lambda_list = [1e-5,1e-2,0.1,1]

mod_A = f.model_eval(X_train,Y_train,M_list,lambda_list)
mod_B = f.model_select(X_val,Y_val,M_list,lambda_list, mod_A)
        
weights, errors, grid = f.train(B, v, M_list, lambda_list)

sse = fun.SSE(X_test,Y_test,1)
w = weights[0,0]
sse(w)
fun.graph_reg(X_train,Y_train,1,w)



w, RSS_final = eval(B[0],B[1], weights[1,0], M = 2, l = 1e-7, plot = True)