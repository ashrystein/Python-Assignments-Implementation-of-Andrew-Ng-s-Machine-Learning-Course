import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as imag
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
from estimateGaussian import estimateGaussian


def multivariateGaussian(X,mu,sigma2):
    k = len(mu)
    sigma2=np.diag(sigma2)
    X = X - mu.T
    p = (1/((2*np.pi)**(k/2)*(np.linalg.det(sigma2)**0.5))) * np.exp(-0.5* np.sum(X @ np.linalg.pinv(sigma2) * X,axis=1))
    return p