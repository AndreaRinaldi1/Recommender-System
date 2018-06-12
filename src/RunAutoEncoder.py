import numpy as np
import pandas as pd
import math
import tensorflow as tf
import os
from scipy.interpolate import interp1d
from autoEncoder import autoEncoder
import IOUtils
import CV
	

def Trainautoencoder(Xtrain, Xtrain_mask, Xtest, Xtest_mask, parameters, doPrediction=False, printIntermediateScore=False, nrTest=58847.0):
	"""Train function for the autoencoder
	@param Xtrain, Xtest: normalized train and testing matrices (of size 10000x1000)
	@param Xtrain_mask, Xtest_mask: the mask corresponding to these matrices.
	Note that Xtest and Xtest_mask don't exist when we do a prediction, since we want to train on the whole data set
	@param parameters: parameters to train this model
	@param doPrediction: True if Xtest and Xtest mask are the indices to submit to Kaggle and we thus do a submission
							False if it is only cross validation on train and validation set
	"""

	#We want to make sure that the format of the matrices correspond to [batch_size, nrFeatures] for tensorflow
	Xtrain = Xtrain.T
	Xtrain_mask = Xtrain_mask.T

	Xtest = Xtest.T
	Xtest_mask = Xtest_mask.T

	nrTrain, nrFeatures = np.shape(Xtrain)

	#Initialize learning rate, and optimizer
	batch = tf.Variable(0)
	batchSize = parameters["batchSize"]
	lr = tf.train.exponential_decay(
		parameters["lr"], 
		batch * batchSize,  
		nrTrain, 
		parameters["lr_decay"], 
		staircase=True)

	optimizer = tf.train.AdamOptimizer(lr)

	model = autoEncoder(parameters, name="autoencoder",
                                 dim=nrFeatures,
                                 optimizer=optimizer,
                                 randSeed=374568,
				 batch=batch)

	with tf.Session() as session:
		session.run(tf.global_variables_initializer())

		nrBatchsPerEpoch = nrTrain//batchSize
		previousLoss=10
		for e in range(parameters["epochs"]):
			print(e)
			indicesToTake = np.arange(nrTrain)
			for i in range(nrBatchsPerEpoch):
				#Choose batchSize indices randomly and remove them from indicesToTake
				chosenTrain = np.random.choice(indicesToTake,batchSize)
				indicesToTake = np.setdiff1d(indicesToTake, chosenTrain)
				XtrainBatch = Xtrain[indicesToTake,:]
				XtrainBatch_mask = Xtrain_mask[indicesToTake,:]	
				if(i==nrBatchsPerEpoch-1):
					preds = model.predict(session,Xtrain)
					loss = math.sqrt((1/(nrTest*1.0))*np.sum(np.square(denormalizeData(Xtest)*np.logical_not(Xtest_mask).astype(int)-denormalizeData(preds)*np.logical_not(Xtest_mask).astype(int))))
					if printIntermediateScore:
						print("Error on validation : ")
						print(loss)
					
					#early stopping
					if loss>previousLoss:
						print("goes up")
						if doPrediction:
							predictions = model.predict(session, Xtrain)
							predictions = denormalizeData(predictions)
							np.save("FullMatrixAuto.npy",predictions.T)
							IOUtil.writeFile(predictions.T)
							print("Predictions made")
						return loss

					previousLoss=loss
					print(previousLoss)
				else:
					cost = model.fit(session, XtrainBatch, XtrainBatch_mask)
		if doPrediction:
			predictions = model.predict(session, Xtrain)
			predictions = denormalizeData(predictions)
			np.save("FullMatrixAuto.npy",predictions.T)
			IOUtil.writeFile(predictions.T)
			print("Predictions made at the very end")
			return loss
					

def normalizeData(X, rangeInf, rangeSup):
	mask = np.ma.array(X, mask=(X==0), dtype = "float32").mask
	interp = interp1d([1,5],[rangeInf,rangeSup])
	height,width = np.shape(X)
	X[X==0]=1
	maskedOnes = np.logical_not(mask*np.ones((height,width))).astype(int)
	return (interp(X)*maskedOnes, mask)


def denormalizeData(X):
	interp = interp1d([-1,1],[1,5])
	return interp(X)

def printScoresParameterSearch(XtrainNorm, Xtrain_mask, XvalNorm, Xval_mask, nrTest):
	parametersFixed={
		"batchSize" : 11,
		"epochs" : 40,
		"hidden_units" : 40,
		"lr" : 0.0005,
		"lr_decay" : 0.9,
		"hideProb": 0.3,
		"gaussianProb": 0.0,
		"gaussianStd": 0.08,
		"mmProb": 0.01, 
		"hiddenFactor": 1.2, 
		"visibleFactor": 0.1, 
		"regularization": 0.1,
	}

	parameters = parametersFixed


	bounds ={
		"gaussianProb" : np.arange(0.0,0.1,0.01),
		"hideProb" : np.arange(0.25,0.35,0.01),
		"mmProb" : np.arange(0,0.1,0.01)  
	}
	results={}
	for key,ran in bounds.items():
		for paramVal in ran:
			parameters=parametersFixed
			parameters[key]=paramVal
			loss = Trainautoencoder(XtrainNorm, Xtrain_mask, XvalNorm, Xval_mask, parameters, printIntermediateScore=True, nrTest=nrVal)
			tf.reset_default_graph()
			results[str(key)+", "+str(paramVal)]=loss
			print(loss)

	print(results)


if __name__ == "__main__":
	parameters={
		"batchSize" : 11,
		"epochs" : 40,
		"hidden_units" : 40,
		"lr" : 0.0005,
		"lr_decay" : 0.89,
		"hideProb": 0.27,
		"gaussianProb": 0.07,
		"gaussianStd": 0.08,
		"mmProb": 0.06, 
		"hiddenFactor": 1.2, 
		"visibleFactor": 0.1, 
		"regularization": 0.1,
	}

	height = 10000
	width = 1000

	#Doing everything on the train dataset
	X = np.load("../data/TrainSet.npy")
	Xtrain, Xval, nrTrain, nrVal = CV.splitNpy(X, height, width, 0.005)
	XtrainNorm, Xtrain_mask = normalizeData(Xtrain, -1,1)
	XvalNorm, Xval_mask = normalizeData(Xval, -1,1)
	#printScoresParameterSearch(XtrainNorm, Xtrain_mask, XvalNorm, Xval_mask, nrVal)
	Trainautoencoder(XtrainNorm, Xtrain_mask, XvalNorm, Xval_mask, parameters, printIntermediateScore=True, nrTest=nrVal, doPrediction=True)












