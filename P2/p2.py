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

X = data[0]
Y = data[1]


def ml_weight(X,Y,M):
    n = X.size
    m = np.zeros((n,M+1))  #desgin matrix
    for i in range(n):
        for j in range(M+1):
            m[i,j] = X[i]**j        
    m_t = np.matrix.transpose(m)
    p1 = npl.inv(np.dot(m_t,m))
    p2 = np.dot(m_t,Y)
    return np.dot(p1,p2)
    
theta = ml_weight(X,Y,0)    

x = np.arange(0,1,.02)

y = np.zeros(len(x))
for i in range(len(theta)):
    y += x**i*theta[i]
    
y2 = np.cos(x*math.pi)+np.cos(x*2*math.pi)

plt.plot(X,Y,'o')
plt.plot(x,y)
plt.plot(x,y2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
