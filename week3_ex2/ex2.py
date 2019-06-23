import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from costFunction import costFunction
from gradient import gradient
from fmintnc import fmintnc
from sigmoid import sigmoid
from predict import predict
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from gradientReg import gradientReg
from plotDecisionBoundary import plotDecisionBoundary


#2
data = pd.read_csv('ex2data2.txt', header=None)
X = data.iloc[:,0:2]
y = data.iloc[:,2]
m = len(y)
data_top = data.head()
print(data_top)


#2.1
mask = y == 1
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
test1 = plt.plot(X[mask][0], X[mask][1] ,'k+',label = 'y = 1')
test2 = plt.plot(X[~mask][0], X[~mask][1] ,'yo',label = 'y = 0')
#plt.legend()
#plt.show()

#2.2
X = mapFeature(X.iloc[:,0],X.iloc[:,1])

#2.3
(m, n) = X.shape
y = y[:, np.newaxis]
theta = np.zeros((n,1))
lam = 0.5    #modify lamda to see the changes
J = costFunctionReg(theta,X,y,lam)
print("J:  " , J)

grad = gradientReg(theta,X,y,lam)
print(grad)

#fmintnc

theta_optimized = opt.fmin_tnc(func = costFunctionReg, x0 = theta.flatten(), fprime = gradientReg,args = (X, y.flatten(), lam))
theta = theta_optimized[0]
print(theta) # theta contains the optimized values

#accurcy
pred = [sigmoid(np.dot(X, theta)) >= 0.5]
acc = np.mean(pred == y.flatten()) * 100
print(acc,'%')


#2.4
u = np.linspace(-1, 1.5, 50)[:,np.newaxis]
v = np.linspace(-1, 1.5, 50)[:,np.newaxis]
z = plotDecisionBoundary(theta,u,v)
print("z:  " , z)

decision_boundary = plt.contour(u[:,0],v[:,0],z,0)
plt.legend()
plt.show()
