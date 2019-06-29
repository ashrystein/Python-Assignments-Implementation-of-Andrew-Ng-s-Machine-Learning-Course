import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt



def randInitializeWeights(L_in, L_out):
    eps = 0.12
    initial_Theta = np.random.rand(L_out,L_in+1)*2*eps-eps
    return initial_Theta