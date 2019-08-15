import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
#from PIL import PIL


def displayData(Theta1):
    img = 0
    fig , subplt = plt.subplots(10,10,figsize=(10,10))
    for i in range(10):
        for j in range(10):
            subplt[i,j].imshow(Theta1[img].reshape((28,28)))
            subplt[i,j].axis('off')
            img+=1
    plt.show()
    return 1