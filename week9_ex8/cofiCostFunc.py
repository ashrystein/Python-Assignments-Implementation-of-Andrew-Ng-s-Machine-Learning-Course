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


def cofiCostFunc(params, Y, R, num_users, num_movies,num_features,lam):
    X = params[:num_movies*num_features].reshape(num_movies,num_features)
    Theta = params[num_movies*num_features:].reshape(num_users,num_features)
    
    predictions =  X @ Theta.T
    err = (predictions - Y)
    J = 1/2 * np.sum((err**2) * R)

    #compute regularized cost function
    reg_X =  lam/2 * np.sum(X**2)
    reg_Theta = lam/2 *np.sum(Theta**2)
    reg_J = J + reg_X + reg_Theta

    # Compute gradient
    X_grad = err*R @ Theta
    Theta_grad = (err*R).T @ X
    grad = np.append(X_grad.flatten(),Theta_grad.flatten())

    # Compute regularized gradient
    reg_X_grad = X_grad + lam*X
    reg_Theta_grad = Theta_grad + lam*Theta
    reg_grad = np.append(reg_X_grad.flatten(),reg_Theta_grad.flatten())


    return J, grad, reg_J, reg_grad