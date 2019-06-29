import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from displayData import displayData
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from backPropagation import backPropagation
from computeNumericalGradient import computeNumericalGradient

def fmincg(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X, y,lam):
    theta_opt = opt.fmin_cg(maxiter = 50, f = nnCostFunction, x0 = initial_nn_params, fprime = backPropagation,
                            args = (input_layer_size, hidden_layer_size, num_labels, X, y.flatten(), lam))
    return theta_opt