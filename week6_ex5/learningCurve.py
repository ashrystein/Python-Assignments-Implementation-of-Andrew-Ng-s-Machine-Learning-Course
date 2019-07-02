import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from costFunctionReg import costFunctionReg
from gradientReg import gradientReg
from trainLinearReg import trainLinearReg


def learningCurve(X, y, Xval, yval, lam):
    m = len(y)

    error_train = np.zeros([m, 1])
    error_val   = np.zeros([m, 1])
    iterations   = np.arange(m)
    for i in range(1,m):
        error_theta_train = trainLinearReg(X[:i,:],y[:i,:],lam)
        error_train[i] = costFunctionReg(error_theta_train[:,np.newaxis],X[:i,:],y[:i,:],0)
        error_val[i] = costFunctionReg(error_theta_train[:,np.newaxis],Xval,yval,0)
    
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.plot(iterations,error_val,'g')
    plt.plot(iterations,error_train,'b')
    plt.show()
    return 1