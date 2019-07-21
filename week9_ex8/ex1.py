import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as imag
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
from estimateGaussian import estimateGaussian
from multivariateGaussian import multivariateGaussian
from visualizeFit import visualizeFit
from selectThreshold import selectThreshold


data = loadmat('ex8data1.mat')
X = data['X']
Xval = data["Xval"]
yval = data['yval']
plt.plot(X[:,0],X[:,1],'bx')
#plt.show()


#1.2
(mu , sig2) = estimateGaussian(X)
p = multivariateGaussian(X,mu,sig2)
#visualizeFit(p,mu,sig2)
#plt.show()

#1.3

pcv = multivariateGaussian(Xval,mu,sig2)

(epi , f1) = selectThreshold(pcv,yval)
outliers = (p < epi)
#plt.scatter(X[outliers,0],X[outliers,1],marker ="o",facecolor="none",edgecolor="r",s=70)
#plt.show()


#1.4
mat2 = loadmat("ex8data2.mat")
X2 = mat2["X"]
Xval2 = mat2["Xval"]
yval2 = mat2["yval"]

mu2, sig2_2 = estimateGaussian(X2)
p2 = multivariateGaussian(X2,mu2,sig2_2)

p2cv = multivariateGaussian(Xval2,mu2,sig2_2)
epsilon2, F1_2 = selectThreshold(p2cv,yval2)

