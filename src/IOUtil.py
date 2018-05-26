import numpy as np
import pandas as pd
from scipy.stats import wishart
import math
import time



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


def predict(Xcompleted, i,j):
	#Return the result here
	return 1

def writeToCSV(Xcompleted):
	df = pd.read_csv("sampleSubmission.csv")
	ids=np.array(df['Id'])
	predictions=np.zeros(np.shape(ids)[0])
	print("Number of predictions to make: " +str(np.shape(ids)[0]))
	start = 0
	j=0
	for i in range(np.shape(ids)[0]):
		row,col=parseId(ids[i])
		predictions[i] = predict(Xcompleted,row,col)
		if(i%10000 == 0):
			if(i!=0):
				print(str(j) +" , time "+str(time.time()-start))
				j+=1
			start = time.time();
	df = pd.DataFrame({'Id':np.ndarray.flatten(ids),'Prediction':np.ndarray.flatten(predictions)})
	df.to_csv("MySubmission.csv",index=False)	


#Example of use:
#height = 10000
#width = 1000
#X=fillMatrix(height,width,0)
#Y = myModel(X)
#writeToCSV(Y)



