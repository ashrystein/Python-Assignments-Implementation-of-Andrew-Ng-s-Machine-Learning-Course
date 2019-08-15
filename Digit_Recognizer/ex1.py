import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score
from displayData import displayData


trainData = pd.read_csv('train.csv')
Xtrain = trainData.iloc[:,1:].values
yTrain = trainData['label'].values
#print(Xtrain.shape)
#displayData(Xtrain)


clf = MLPClassifier(solver='adam',hidden_layer_sizes=(40,),activation="relu")
fit = clf.fit(Xtrain, yTrain.ravel())
print(fit)
NNTrainpredictions = clf.predict(Xtrain)
print("Training Accuracy NN:",(accuracy_score(NNTrainpredictions,yTrain.ravel()))*100,"%")


#plt.plot(clf.loss_curve_)
#plt.show()


testData = pd.read_csv('test.csv')
Xtest = testData.iloc[:,:].values


SampleData = pd.read_csv('sample_submission.csv')
SampleID = SampleData.iloc[:,0]

NNTestPrediction = clf.predict(Xtest)

label = pd.DataFrame(NNTestPrediction)

print(label) 
print(SampleID) 



result = pd.concat([SampleID, label], axis=1)
result.columns = ['ImageId', 'Label']
result.to_csv('submission.csv', index=False)
