import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as imag
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
from estimateGaussian import estimateGaussian
from multivariateGaussian import multivariateGaussian


def visualizeFit(p,mu,sigma2):
    X1,X2 = np.meshgrid(np.linspace(0,35,num=70),np.linspace(0,35,num=70))
    p2 = multivariateGaussian(np.hstack((X1.flatten()[:,np.newaxis],X2.flatten()[:,np.newaxis])), mu, sigma2)
    contour_level = 10**(np.arange(-20., 1, 3))
    plt.contour(X1,X2,p2[:,np.newaxis].reshape(X1.shape),contour_level)
    plt.xlim(0,35)
    plt.ylim(0,35)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    return