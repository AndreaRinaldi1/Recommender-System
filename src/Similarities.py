import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import datetime
import time


def parseId(stringId):
    splits = stringId.split("_")
    return int(splits[0][1:])-1,int(splits[1][1:])-1


def writeFile(X):
    df = pd.read_csv('sampleSubmission.csv')
    ids=np.array(df['Id'])
    predictions=np.zeros(np.shape(ids)[0])
    for i in range(np.shape(ids)[0]):
        row,col=parseId(ids[i])
        predictions[i] = X[row,col]
    df = pd.DataFrame({'Id':np.ndarray.flatten(ids),'Prediction':np.ndarray.flatten(predictions)})
    now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv("mySubmission"+now+".csv",index=False)


def prediction(ratings, similarity, type):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        mask = np.nonzero(ratings)

        ratings = ratings[mask]
        similarity = similarity[mask]
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


def predict(userItemMatrix, similarityMatrix, user, item):
    simCoeff = similarityMatrix[item, :]
    userRatings = userItemMatrix[user, :]  # all the ratings given by the user u

    mask = np.nonzero(userRatings)

    valSimCoeff = simCoeff[mask]
    valUrat = userRatings[mask]

    num = np.sum(valSimCoeff * valUrat)
    dem = np.sum(np.absolute(valSimCoeff))
    res = num / dem
    return res


def predictTot(userItemMatrix, similarityMatrix):
    row = userItemMatrix.shape[0]
    col = userItemMatrix.shape[1]

    print("I am starting the prediction")

    for i in range(row):

        if (i % 1000 == 0):
            print("I am the user:" + str(i))
        for j in range(col):
            prediction = predict(userItemMatrix, similarityMatrix, i, j)
            userItemMatrix[i][j] = prediction
    return userItemMatrix


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


def pearsonSim(userItemMatrix):
    r = np.repeat(range(userItemMatrix.shape[1]), userItemMatrix.shape[1])
    t = np.tile(range(userItemMatrix.shape[1]), userItemMatrix.shape[1])
    pearMatrix = np.zeros(len(r))
    for i in range(len(r)):

        if i % 10000 == 0:
            print(i)

        ridx = r[i]
        tidx = t[i]

        item1 = userItemMatrix[:, ridx]
        item2 = userItemMatrix[:, tidx]
        pearson_sim = pearsonr(item1, item2)[0]
        pearMatrix[i] = pearson_sim

    pearMatrix = np.reshape(pearMatrix, (n_movies, n_movies))
    np.save("similarityMatrix", pearMatrix)
    return pearMatrix



n_users = 10000
n_movies = 1000

train_data = np.load("TrainSet.npy")

#compute the similarity(cosine)
#user_similarity = pairwise_distances(train_data, metric='cosine')
#item_similarity = pairwise_distances(train_data.T, metric='cosine')
#compute the similarity(pearson)
pearSim = pearsonSim(train_data)

#2) compute the prediction (cosine)
#user_prediction = prediction(train_data, user_similarity, 'user')
#item_prediction = predictTot(train_data, item_similarity)
#2) compute the prediction (pearson)
item_pear_prediction = predictTot(train_data, pearSim )


item_pear_prediction = np.clip(item_pear_prediction, 1, 5)
now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
np.save("item-itemPearson" + now + "", item_pear_prediction)
writeFile(item_pear_prediction)

#item_prediction = np.clip(item_prediction, 1, 5)
#now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#np.save("item-itemDot" + now + "", item_prediction)
#writeFile(item_prediction)