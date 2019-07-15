import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids


def runkMeans(X,initial_centroids,max_iters,K):
    centroids = initial_centroids
    centroid_history = []
    idx = None
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroid_history.append(centroids)
        centroids = computeCentroids(X,idx,K)
        plotProgresskMeans(i,centroid_history , K)
    
    plt.scatter(X[:,0],X[:,1],c=idx,marker='o')
    plt.show()
    
    return centroids , idx


def plotProgresskMeans(i,centroid_history,K):
    for k in range(K):
        current = np.stack([c[k, :] for c in centroid_history[:i+1]], axis=0)
        plt.plot(current[:, 0], current[:, 1],'-Xk')
    return