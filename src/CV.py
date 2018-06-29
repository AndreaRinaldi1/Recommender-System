import numpy as np
import pandas as pd
import os
import IOUtils
import math
import random


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
		row,col=IOUtils.parseId(ids[i])
		Xtrain[row,col] = pred[i]
		
	indicesTest = np.setdiff1d(np.arange(nrRatings), indicesTrain)
	for i in indicesTest:
		row,col=IOUtils.parseId(ids[i])
		Xtest[row,col] = pred[i]

	if os.path.exists("temporary.csv"):
		os.remove("temporary.csv")
	return Xtrain,Xtest,nrTrain, nrTest


def splitNpy(X, height, width, splitPercentage):
	nonzeros = np.nonzero(X)
	ids = np.empty(np.shape(nonzeros[0])[0], dtype="U12") 
	predictions = np.empty(np.shape(nonzeros[0])[0], dtype="i")
	for i in range(np.shape(nonzeros[0])[0]):
		h = nonzeros[0][i]
		w = nonzeros[1][i]
		idString = "r"+str(h+1)+"_c"+str(w+1)
		pred = X[h,w]
		ids[i] = idString
		predictions[i] = pred
	
	df = pd.DataFrame({'Id':ids,'Prediction':predictions})
	df.to_csv("temporary.csv",index=False)	
	return splitSetCSV(splitPercentage,height,width, inp="temporary.csv")


def hide_values(Ind):
    x = Ind[np.argsort(Ind[:, 0])]
    index = 0
    hidden_values = []
    while index < len(x):
        count = 0.0
        while index+1 < len(x) and x[index+1,0] == x[index,0]:
            count += 1.0
            index += 1
        #print(count)
        arg_hidden_values = [random.randint(0, count) for _ in range(math.ceil(count/10.0))]
        #print([random.randint(0, count) for _ in range (math.ceil(count/10.0)) ])
        for i in arg_hidden_values:
            hidden_values.append(x[index-i])
        index += 1
    return hidden_values


def create_training_set(Ind, full_matrix, unknown_value):
    hidden_values = hide_values(Ind)

    training_matrix = full_matrix.copy()
    for row, col in hidden_values:
        training_matrix[row, col] = 0

    remaining_values = []
    for i in range(0, len(full_matrix)):
        for j in range(0, len(full_matrix[0])):
            if training_matrix[i, j] != unknown_value:
                remaining_values.append([i, j])

    return training_matrix, remaining_values, hidden_values
