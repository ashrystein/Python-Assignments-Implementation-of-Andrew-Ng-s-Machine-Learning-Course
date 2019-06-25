import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

def displayData(X):
    fig , subplt = plt.subplots(10,10,figsize=(10,10))
    for i in range(10):
        for j in range(10):
            subplt[i,j].imshow(X[np.random.randint(X.shape[0])].reshape((20,20), order = 'F'))
            subplt[i,j].axis('off')
    plt.show()
    return 1