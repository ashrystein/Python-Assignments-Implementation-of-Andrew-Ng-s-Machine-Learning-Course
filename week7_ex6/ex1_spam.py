import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
from gaussianKernel import gaussianKernel
from dataset3Params import dataset3Params
from processEmail import processEmail
from emailFeatures import emailFeatures

#2.1
email_contents = open("emailSample1.txt","r").read()
vocabList =  open("vocab.txt","r").read()



#2.1.1
vocabList=vocabList.split("\n")[:-1]
vocabList_d={}
for ea in vocabList:
    value,key = ea.split("\t")[:]
    vocabList_d[key] = value

word_indices = processEmail(email_contents , vocabList_d)

#2.2

featureVector = emailFeatures(word_indices)
#print(np.sum(featureVector))

#2.3
spamTrainData = loadmat('spamTrain.mat')
#print(spamTrainData)

X = spamTrainData['X']
y = spamTrainData['y']

print('>   ' ,spamTrainData)

C =0.1
kernel ="linear"
classifier = SVC(C = C,kernel = kernel)
classifier.fit(X,y.ravel())
print("Training Accuracy:",(classifier.score(X,y.ravel()))*100,"%")


spamDataTest = loadmat('spamTest.mat')
Xtest = spamDataTest['Xtest']
ytest = spamDataTest['ytest']
classifier.predict(Xtest)
print("Testing Accuracy:",(classifier.score(Xtest,ytest.ravel()))*100,"%")


#2.4
weights = classifier.coef_[0]
weights_col = np.hstack((np.arange(1,1900).reshape(1899,1),weights.reshape(1899,1)))
df = pd.DataFrame(weights_col)
df.sort_values(by=[1],ascending = False,inplace=True)

for i in df[0][:15]:
    for key, value in vocabList_d.items():
        if (str(int(i)) == value):
            print(key,'    ',df[1][int(value)-1])


