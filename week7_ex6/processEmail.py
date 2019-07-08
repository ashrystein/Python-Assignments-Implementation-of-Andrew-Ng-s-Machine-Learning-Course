import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
from gaussianKernel import gaussianKernel
from dataset3Params import dataset3Params
import re
from nltk.stem import PorterStemmer



def processEmail(email_contents , vocab):
    email_contents = email_contents.lower()
    email_contents = re.sub("[0-9]+","number",email_contents)
    email_contents = re.sub("(http|https)://[^\s]*","httpaddr",email_contents)
    email_contents = re.sub("[^\s]+@[^\s]+","emailaddr",email_contents)
    email_contents = re.sub("[$]+","dollar",email_contents)
    specialChar = ["<","[","^",">","+","?","!","'",".",",",":"]
    for char in specialChar:
        email_contents = email_contents.replace(str(char),"")
    email_contents = email_contents.replace("\n"," ")

    ps = PorterStemmer()
    email_contents = [ps.stem(token) for token in email_contents.split(" ")]
    email_contents= " ".join(email_contents)

    word_indices=[]

    for word in email_contents.split():
        if(len(word) > 1 and word in vocab):
            word_indices.append(int(vocab[word]))

    return word_indices