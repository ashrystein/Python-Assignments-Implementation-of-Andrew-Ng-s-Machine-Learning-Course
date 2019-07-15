import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as imag
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids
from featureNormalize import featureNormalize
from pca import pca
from projectData import projectData
from recoverData import recoverData
from displayData import displayData


#2.1
data = loadmat('ex7data1.mat')
X = data['X']
#plt.plot(X[:,0],X[:,1],'bo')
#plt.show()


#2.2
(X_norm, mu, sigma)= featureNormalize(X)
(U, S, V) = pca(X_norm)
#for i in range(2):
    #plt.arrow(mu[0], mu[1], 1.5 * S[i]*U[0, i], 1.5 * S[i]*U[1, i])
#plt.show()


#2.3.1
K = 1
Z = projectData(X_norm, U, K)

X_rec = recoverData(Z , U , K)
#print(X_rec)


#2.3.3
#plt.plot(X_rec[:,0],X_rec[:,1],'ro')
#plt.plot(X_norm[:,0],X_norm[:,1],'bo')

#for i in range(X_norm.shape[0]):
    #plt.plot([X_norm[i][0], X_rec[i][0]], [X_norm[i][1], X_rec[i][1]], '--k', lw=1)

#plt.show()


#2.4
data = loadmat('ex7faces.mat')
X = data['X']
#displayData(X,32,32)


#2.4.1
(X_norm, mu, sigma)= featureNormalize(X)
(U, S, V) = pca(X_norm)
#displayData(U.T,32,32)


K = 100
Z = projectData(X_norm, U, K)

X_rec = recoverData(Z , U , K)

displayData(X,32,32)
displayData(X_rec,32,32)
