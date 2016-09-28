# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 00:40:15 2016

@author: loredp
"""

import sklearn.linear_model as skls
import lassoData
import pylab as pl
import functions as fun

train = lassoData.lassoTrainData()
val = lassoData.lassoValData()
test = lassoData.lassoTestData()

true_w = pl.loadtxt('lasso_true_w.txt') #True value for the data

#Step 1: transform data

alpha = 0.2
M = 12


X = train[0]
Xm = fun.phi(train[0],M,fun.basis_sin)
Y = train[1]
Xc = Xm-Xm.mean(axis=0)
Yc = Y - Y.mean()
lasso = skls.Lasso(alpha=alpha, fit_intercept = False)
lasso.fit(Xc,Yc)

#alpha 0.2

#Compare with ridge

wr = fun.ridge(X,Y,12,0.0,basis_sin) 
sse = fun.SSE(X,Y,wr,M, l = alpha)
s = sse(wr)

fun.evaluate(X,Y,12,wr, alpha, plot = True)


#OlS

w_ols = ml_weight(X, Y, M = 12, basis = basis_sin)
