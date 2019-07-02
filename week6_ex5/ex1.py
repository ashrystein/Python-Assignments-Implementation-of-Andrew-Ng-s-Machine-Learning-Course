import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from costFunctionReg import costFunctionReg
from gradientReg import gradientReg
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from polyfit import polyfit


#1
data = loadmat('ex5data1.mat')
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']
#print(X.shape , Xtest.shape  , Xval.shape)


#1.1
plt.xlabel('Change in water level')
plt.ylabel('Water flowing out of the dam')
plt.plot(X,y,'rx')
#plt.show()

#1.2,1.3
m = len(y)
ones = np.ones((m,1))
X = np.hstack((ones, X))
lam = 0
theta = np.ones([2,1])
J = costFunctionReg(theta,X,y,lam)
print(J)

grad = gradientReg(theta,X,y,lam)
print(grad)


#1.4
lam = 0
thetaOpt = trainLinearReg(X,y,lam)
print(thetaOpt)
hypothesis = np.dot(X,thetaOpt)
#plt.plot(X[:,1],hypothesis,'b')
#plt.show()


#2.1
lam =1
#learningCurve(X, y, Xval, yval, lam)

#3
p = 8
Xpoly = polyFeatures(X,p)
Xpoly, mu, sigma = featureNormalize(Xpoly)
Xpoly = np.hstack((ones, Xpoly))


mtest = len(ytest)
ones = np.ones((mtest,1))
Xtest = np.hstack((ones, Xtest))
X_poly_test = polyFeatures(Xtest,p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.hstack((ones, X_poly_test))


mval = len(yval )
ones = np.ones((mval,1))
Xval  = np.hstack((ones, Xval ))
X_poly_val  = polyFeatures(Xval,p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val  = np.hstack((ones, X_poly_val ))


lam = 0
thetaOpt = trainLinearReg(Xpoly,y,lam)

polyfit(np.min(Xpoly),np.max(Xpoly),mu, sigma, thetaOpt, p)

lam =0
learningCurve(Xpoly, y, X_poly_val, yval, lam)

plt.show()