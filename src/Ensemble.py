import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import IOUtils



def ensemble(known_ids, idsToWrite):
	path = "../../CIL_results/results/"
	files = [f for f in listdir(path) if isfile(join(path, f))]

	nrFiles = 0
	model_bias = getModelBias()
	for file in files:
		matrix = np.load(path + file)
		train = []
		pred = []
		print(str(file))
		for row, col in known_ids:
			train.append(matrix[row, col])
		for row2, col2 in idsToWrite:
			pred.append(matrix[row2, col2])
		diff = np.abs(np.round(train) - train)
		diffTo05 = (np.maximum(0.01, 0.5 - diff))**0.1
		weight = diffTo05 * model_bias[file]
		if nrFiles == 0:
			train_matrix = train
			pred_matrix = pred
			models_weights = weight
		else:
			train_matrix = np.vstack((train_matrix, train))
			pred_matrix = np.vstack((pred_matrix, pred))
			models_weights = np.vstack((models_weights, weight))
		nrFiles+=1

	weights_sum = np.sum(models_weights, axis=0)

	return train_matrix.T, pred_matrix.T, weights_sum


def regression(train_matrix, pred_matrix, groundTruth):
	mlp = MLPRegressor()
	params = {"hidden_layer_sizes": [(25,), (50,), (75,), (100,), (125,)], "activation":["relu"], "solver":["adam"],
			  "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01], "learning_rate": ["invscaling"]}
	grid = GridSearchCV(estimator=mlp, param_grid=params, iid=False, verbose=3, n_jobs=-1, scoring="neg_mean_squared_error")
	grid.fit(train_matrix, groundTruth)
	final_prediction = grid.predict(pred_matrix)
	print(grid.best_score_)
	print(grid.best_params_)
	return final_prediction


def getModelBias():
	exp = 50
	return {'NPCA.npy': (1/1.0264)**exp, 'BPMF.npy': (1/0.99695)**exp, 'RSVD.npy': (1/0.98787)**exp,
			'Ridge.npy': (1/1.0264)**exp, 'item-itemPearson.npy': (1/1.05)**exp,
			'KMeans.npy': (1/1.06)**exp, 'Autoencoder.npy': (1/0.996)**exp}


def getIdsToWrite():
	df = pd.read_csv("../data/sampleSubmission.csv")
	ids=np.array(df['Id'])
	idsToWrite = np.zeros((len(ids), 2))
	for i in range(np.shape(ids)[0]):
		row,col=IOUtils.parseId(ids[i])
		idsToWrite[i] = row, col
	return idsToWrite.astype(int)


def groundTruth():
	X = np.load("../data/ValidationSet.npy")
	rows, cols = np.where(X != 0)
	Ind = np.zeros((rows.size, 2), dtype=np.int)
	pred = np.zeros(rows.size)
	for i in range(rows.size):
		Ind[i] = np.array([rows[i], cols[i]])
		pred[i] = X[rows[i], cols[i]]
	return Ind.astype(int), pred


def main():
	known_ids, groundT = groundTruth()
	idsToWrite = getIdsToWrite()
	train_matrix, pred_matrix, weights = ensemble(known_ids, idsToWrite)
	final_prediction = regression(train_matrix, pred_matrix, groundT)
	final_prediction = np.clip(final_prediction, 1, 5)
	IOUtils.writeFileEnsemble(final_prediction)



if __name__ == '__main__':
	main()

