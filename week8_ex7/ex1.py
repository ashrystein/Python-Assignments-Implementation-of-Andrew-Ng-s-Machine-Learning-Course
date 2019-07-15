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


#1
data = loadmat('ex7data2.mat')
X = data['X']
#print(X)
#plt.plot(X[:,0],X[:,1],'k+')
#plt.show()

#1.1.1
K = 3
initial_centroids = np.array([[3,3],[6,2],[8,5]])
idx = findClosestCentroids(X,initial_centroids)
#print(idx)

#1.1.2
centroids = computeCentroids(X,idx,K)
#print(centroids)

#1.2
max_iters = 10
#(centroids , idx) = runkMeans(X,initial_centroids,max_iters,K)

#1.3

rn_centroids = kMeansInitCentroids(X,K)
#print(rn_centroids)

#1.4.1
K = 16
max_iters = 10
img = imag.imread('bird_small.png')

img/=255
X = img.reshape(-1, 3)
initial_centroids = kMeansInitCentroids(X, K)
(centroids , idx) = runkMeans(X,initial_centroids,max_iters,K)
X_reshaped = idx.reshape(img.shape[0],img.shape[1])

X_recovered = np.zeros((X_reshaped.shape[0],X_reshaped.shape[1],3))

for i in range(X_reshaped.shape[0]):
    for j in range(X_reshaped.shape[1]):
        X_recovered[i][j] = centroids[X_reshaped[i][j]-1]

orig = imag.imread('bird_small.png')
plt.imshow(orig)
plt.show()
plt.imshow(X_recovered*255)
plt.show()

