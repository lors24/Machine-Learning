# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 00:40:15 2016

@author: loredp
"""

import sklearn.linear_model as skls
import lassoData
import pylab as pl
import functions as fun
import matplotlib.pyplot as plt
import numpy as np

train = lassoData.lassoTrainData()
val = lassoData.lassoValData()
test = lassoData.lassoTestData()

true_w = pl.loadtxt('lasso_true_w.txt') #True value for the data

#Step 1: transform data

X = train[0]
Y = train[1]

(Xc,Yc) = fun.center_data(X,Y)

alpha = 0.2
fig = plt.figure()
fig.add_subplot(121)
w1 = fun.compare(X,Y,M=12,alpha=alpha,basis = fun.basis_sin)
plt.bar(range(13),w1[0], color = 'teal')
plt.title('LASSO')
fig.add_subplot(122)
plt.bar(range(13),w1[1], color = 'purple')
plt.title('ridge')
plt.show()

fig = plt.figure()
fig.add_subplot(121)
plt.plot()
fig.add_subplot(122)

plt.bar(range(13),w1[0], color = 'teal')
plt.title('lambda' = alpha)
plt.plot(range(10),np.ones(10), '--r')
plt.xticks(np.arange(9) + 0.35, ('0','1','2','3','4','5','6','7','8'))
plt.title('weight vector for M=8')
plt.show()

#alpha 0.2

#Compare with ridgefu

alpha = 0.2

#wr = fun.ridge(Xc,Yc,1,alpha) 
#ridge = skls.Ridge(alpha = alpha, fit_intercept = False)
#ridge.fit(Xc,Yc)
#w_r = ridge.coef_

#OlS

w_ols = ml_weight(Xc, Yc, M = 12, basis = basis_sin)

#compare


#Ridge from skl and 

w_ridge = fun.ridge(X,Y,M=12,basis = basis_sin, alpha = 0)

#linear = skl.LinearRegression(fit_intercept = False)
#linear.coef_
#linear.fit(Xc,Yc)

#plt.bar(np.arange(1,14),w_ols, color = 'cyan')
#plt.xlim(1,14)
#plt.xticks(np.linspace(1,13,1))