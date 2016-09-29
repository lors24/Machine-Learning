# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:29:48 2016

@author: loredp
"""

from loadFittingDataP2 import getData
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
import functions as fun
#from matplotlib import gridspec

data = getData(ifPlotData=True)

X = data[0][:,None]
Y = data[1][:,None] 

#2.1 

for i in [0,1,3,5,10]:
    w = fun.ml_weight(X,Y,i)
    fun.graph_reg(X,Y,i,w,f=fun.q2)

#Plot 1 

x_plot = np.linspace(0,1,100)
plt.plot(X,Y,'o', linewidth = 2)
y_true= fun.q2(x_plot) 
plt.plot(x_plot,y_true, '--',color = 'purple', linewidth = 2, label = 'true function')
for i in [1,2,5,10]:
    w = fun.ml_weight(X,Y,i)
    y_plot = np.dot(fun.phi(x_plot,i),w)  
    plt.plot(x_plot,y_plot,label = 'M = ' + str(i), linewidth = 1.5)
plt.legend(loc = 1, fontsize = 10)
plt.xlabel('x')
plt.ylabel('y')

#2.2 

#Check gradient

fun.check_grad(X,Y,M = 11,t = 0)

#Plot error


SSE_s = np.zeros(11)
for i in range(11):
    SSE_s[i] = fun.eval_reg(X, Y, i, basis = fun.poli)
    
fig = plt.figure()
fig.add_subplot(121)
plt.plot(range(11),SSE_s, color = 'green', linewidth = 2)
plt.xlabel('M')
plt.ylabel('SSE')
plt.title('SSE for M=0 to M=10')
fig.add_subplot(122)
plt.plot(np.arange(2,11),SSE_s[2:], linewidth = 2)
plt.xlabel('M')
plt.title('SSE for M=2 to M=10')
plt.show()
    





alpha = 1e-3

def plot_ridge(alpha):
    x_plot = np.linspace(0,1,100)
    plt.plot(X,Y,'o', linewidth = 2)
    y_true= fun.q2(x_plot) 
    plt.plot(x_plot,y_true, '--',color = 'purple', linewidth = 2, label = 'true function')
    for i in [1,2,5,10]:
        w = fun.ridge(X,Y,i,alpha)
        y_plot = np.dot(fun.phi(x_plot,i),w)  
        plt.plot(x_plot,y_plot,label = 'M = ' + str(i), linewidth = 1.5)
    #plt.legend(loc = 1, fontsize = 10)
    plt.xlabel('x')
    plt.xticks(np.linspace(0,1,5))
    plt.ylabel('y')
    plt.title('alpha = ' + str(alpha))


fig = plt.figure()


fig.add_subplot(141)
plot_ridge(1e-7)

fig.add_subplot(142)
plot_ridge(1e-4)

fig.add_subplot(143)
plot_ridge(1e-1)

fig.add_subplot(144)
plot_ridge(1)
plt.figure(figsize=(3,4))
plt.show()







w,k = fun.gradient_descent(fSSE, gSSE, w3*0 ,0.05,1e-07, maxiter = 100000)
#w2, res = fun.evaluate(X,Y,8,basis = fun.cos, plot = True, f = fun.q2)

#P3

M = 2
alpha = 0
wr = fun.ridge(X,Y,M,alpha ) 
#sse = fun.SSE(X,Y,wr,M,l)
#s = sse(wr)
#fun.evaluate(X,Y,M,wr, l, plot = True, f = fun.q2)

w_1,k_1,f_delta_1 = fun.gradient_descent(fSSE, gSSE, w3*0 ,0.01,1e-6,maxiter=2000)
w_2,k_2,f_delta_2 = fun.gradient_descent(fSSE, gSSE, w3*0 ,0.05,1e-6,maxiter=2000)
w_3,k_3,f_delta_3 = fun.gradient_descent(fSSE, gSSE, w3*0 ,0.001,1e-6,maxiter=2000)
#w_4,k_4,f_delta_4 = gradient_descent(fSSE, gSSE, w3*0 ,0.1,1e-6,maxiter=2000)



plt.plot(f_delta_3, label = 'alpha = 0.001')
plt.plot(f_delta_1, label = 'alpha = 0.01')
plt.plot(f_delta_2, label = 'alpha = 0.05')
plt.ylabel('SSE')
plt.xlabel('#iter')
plt.legend()
plt.title('M=3, w_0 = [0,0,0,0]')

w_4,k_4,f_delta_4 = gradient_descent(fSSE, gSSE, w3+1 ,0.001,1e-6,maxiter=2000)
w_5,k_5,f_delta_5 = gradient_descent(fSSE, gSSE, w3+1 ,0.01,1e-6,maxiter=2000)
w_6,k_6,f_delta_6 = gradient_descent(fSSE, gSSE, w3+1 ,0.05,1e-6,maxiter=2000)

plt.plot(f_delta_4[0:100], label = 'alpha = 0.001')
plt.plot(f_delta_5[0:100], label = 'alpha = 0.01')
plt.plot(f_delta_6[0:100], label = 'alpha = 0.05')
plt.ylabel('SSE')
plt.xlabel('#iter')
plt.legend()
plt.title('M=3, w_0 = [0,0,0,0]')

w_0 = np.ones((4,1))*100

w_7,k_7,f_delta_7 = gradient_descent(fSSE, gSSE, w_0 ,0.05,1e-6,maxiter=2000)
w_8,k_8,f_delta_8 = gradient_descent(fSSE, gSSE, w_0 ,0.01,1e-6,maxiter=2000)
w_9,k_9,f_delta_9 = gradient_descent(fSSE, gSSE, w_0 ,0.001,1e-6,maxiter=2000)
