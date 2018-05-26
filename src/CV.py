import numpy as np
import pandas as pd
import math



def parseId(stringId):
	splits = stringId.split("_")
	return int(splits[0][1:])-1,int(splits[1][1:])-1

def fillMatrix(height,width,val):
	X = np.ones(shape=(height,width))*val
	df = pd.read_csv("data_train.csv")
	ids=np.array(df['Id'])
	pred=np.array(df['Prediction'])
	for i in range(np.shape(ids)[0]):
		row,col=parseId(ids[i])
		X[row,col]=pred[i]
	return X


def splitSet(validationPercentage, height,width):

	df = pd.read_csv("data_train.csv")
	ids=np.array(df['Id'])
	pred=np.array(df['Prediction'])
	nrRatings = np.shape(ids)[0]
	nrTrain = (int)(nrRatings*(1-validationPercentage))
	indicesTrain = np.random.choice(nrRatings,nrTrain,replace=False)
	Xtrain = np.zeros((height,width))
	Xtest = np.zeros((height,width))

	nrTrain=0
	for i in indicesTrain:
		row,col=parseId(ids[i])
		Xtrain[row,col] = pred[i]
		nrTrain+=1
		
	nrTest=0
	indicesTest = np.setdiff1d(np.arange(nrRatings), indicesTrain)
	for i in indicesTest:
		row,col=parseId(ids[i])
		Xtest[row,col] = pred[i]
		nrTest+=1

	return Xtrain,Xtest, nrTrain, nrTest


def evaluate(Xcompleted, Xtest):
	nonzeros = np.nonzero(Xtest)
	squareSum = 0
	nrRatings = 0
	for i in range(np.shape(nonzeros[0])[0]):
		h = nonzeros[0][i]
		w = nonzeros[1][i]
		squareSum+=(Xtest[h,w]-Xcompleted[h,w])**2
		nrRatings+=1
	return math.sqrt(squareSum*1.0/nrRatings)



#Example of usage:
height = 10000
width = 1000
splitPercentage = 0.05
Xtrain, Xtest, nrTrain, nrTest= splitSet(splitPercentage,height,width)
print("Number of train: "+str(nrTrain))
print("Number of test: "+str(nrTest))
np.save("TrainSet.npy",Xtrain)
np.save("ValidationSet.npy",Xtest)

#Xcomplete = myModel(Xtrain)
#print(evaluate(Xcomplete, Xtest))



