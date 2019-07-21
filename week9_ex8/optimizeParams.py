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
from cofiCostFunc import cofiCostFunc
from os.path import join
from normalizeRatings import normalizeRatings




def optimizeParams(initial_parameters,Y,R,num_users,num_movies,num_features,alpha,num_iters,Lambda):
    X = initial_parameters[:num_movies*num_features].reshape(num_movies,num_features)
    Theta = initial_parameters[num_movies*num_features:].reshape(num_users,num_features)
    J_history =np.zeros((num_iters,2))


    for i in range(num_iters):
        params = np.append(X.flatten(),Theta.flatten())
        J, grad, reg_J, reg_grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda)
        X_grad = reg_grad[:num_movies*num_features].reshape(num_movies,num_features)
        Theta_grad = reg_grad[num_movies*num_features:].reshape(num_users,num_features)
        X = X - (alpha * X_grad)
        Theta = Theta - (alpha * Theta_grad)
        J_history[i][0] = reg_J
        J_history[i][1] = i

    paramsFinal = np.append(X.flatten(),Theta.flatten())
    return paramsFinal , J_history