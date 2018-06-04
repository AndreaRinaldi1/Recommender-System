import numpy as np
import pandas as pd
import math
import os


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


def splitSetCSV(validationPercentage, height,width, inp="../data/data_train.csv"):

	df = pd.read_csv(inp)
	ids=np.array(df['Id'])
	pred=np.array(df['Prediction'])
	nrRatings = np.shape(ids)[0]
	nrTrain = (int)(nrRatings*(1-validationPercentage))
	nrTest = nrRatings-nrTrain
	indicesTrain = np.random.choice(nrRatings,nrTrain,replace=False)
	Xtrain = np.zeros((height,width))
	Xtest = np.zeros((height,width))

	for i in indicesTrain:
		row,col=parseId(ids[i])
		Xtrain[row,col] = pred[i]
		
	indicesTest = np.setdiff1d(np.arange(nrRatings), indicesTrain)
	for i in indicesTest:
		row,col=parseId(ids[i])
		Xtest[row,col] = pred[i]

	if os.path.exists("temporary.csv"):
   		os.remove("temporary.csv")
	return Xtrain,Xtest,nrTrain, nrTest


def splitNpy(X, height, width, splitPercentage):
	nonzeros = np.nonzero(X)
	ids = np.empty(np.shape(nonzeros[0])[0], dtype="U12") 
	predictions = np.empty(np.shape(nonzeros[0])[0], dtype="i")
	for i in range(np.shape(nonzeros[0])[0]):
		if(i%100000 == 0):
			print(i)
		h = nonzeros[0][i]
		w = nonzeros[1][i]
		idString = "r"+str(h+1)+"_c"+str(w+1)
		pred = X[h,w]
		ids[i] = idString
		predictions[i] = pred
	
	df = pd.DataFrame({'Id':ids,'Prediction':predictions})
	df.to_csv("temporary.csv",index=False)	
	return splitSetCSV(splitPercentage,height,width, inp="temporary.csv")


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





