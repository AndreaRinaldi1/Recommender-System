import numpy as np
import pandas as pd
import time
import datetime

complete_trainset = "../data/data_train.csv"
training_set = "../data/TrainSet.npy"
sample_submission = "../data/sampleSubmission.csv"


def parseId(stringId):
	splits = stringId.split("_")
	return int(splits[0][1:])-1,int(splits[1][1:])-1

def fillMatrix(height,width,val):
	X = np.ones(shape=(height,width))*val
	df = pd.read_csv(complete_trainset)
	ids=np.array(df['Id'])
	pred=np.array(df['Prediction'])
	for i in range(np.shape(ids)[0]):
		row,col=parseId(ids[i])
		X[row,col]=pred[i]
	return X

def initialization():
    full_matrix = np.load(training_set)
    rows, cols = np.where(full_matrix != 0)
    Ind = np.zeros((rows.size, 2), dtype=np.int)
    for i in range(rows.size):
        Ind[i] = np.array([rows[i], cols[i]])
    return full_matrix, Ind


def writeFileEnsemble(predictions):
	df = pd.read_csv(sample_submission)
	ids=np.array(df['Id'])
	df = pd.DataFrame({'Id': np.ndarray.flatten(ids), 'Prediction': np.ndarray.flatten(predictions)})
	df.to_csv("EnsembleSubmission.csv", index=False)


def writeFile(X):
    df = pd.read_csv(sample_submission)
    ids=np.array(df['Id'])
    predictions=np.zeros(np.shape(ids)[0])
    for i in range(np.shape(ids)[0]):
        row,col=parseId(ids[i])
        predictions[i] = X[row,col]
    df = pd.DataFrame({'Id':np.ndarray.flatten(ids),'Prediction':np.ndarray.flatten(predictions)})
    now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv("mySubmission"+now+".csv",index=False)







