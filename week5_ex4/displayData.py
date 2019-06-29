import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

def displayData(Theta1):
    img = 0
    fig , subplt = plt.subplots(5,5,figsize=(5,5))
    for i in range(5):
        for j in range(5):
            subplt[i,j].imshow(Theta1[img][:-1].reshape((20,20), order = 'F'))
            subplt[i,j].axis('off')
            img+=1
    plt.show()
    return 1