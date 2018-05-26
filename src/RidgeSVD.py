import numpy as np
import pandas as pd
import datetime
import time


def get_rating(user, movie, U, ZT, b, b_u, b_m_weighted):
    return b_u[user] + b_m_weighted[movie] + U[user, :].dot(ZT[:, movie])


def getFullMatrix(A, Ind, U, ZT, b, b_u, b_m_weighted):
    fullMatrix = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            fullMatrix[i, j] = get_rating(i, j,U, ZT, b, b_u, b_m_weighted)
    return fullMatrix


def writeFile(X):
    df = pd.read_csv("sampleSubmission.csv")
    ids=np.array(df['Id'])
    predictions=np.zeros(np.shape(ids)[0])
    for i in range(np.shape(ids)[0]):
        row,col=parseId(ids[i])
        predictions[i] = X[row,col]
    df = pd.DataFrame({'Id':np.ndarray.flatten(ids),'Prediction':np.ndarray.flatten(predictions)})
    now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv("mySubmission"+now+".csv",index=False)


def parseId(stringId):
    splits = stringId.split("_")
    return int(splits[0][1:])-1,int(splits[1][1:])-1


def initialization():
    A = np.load("TrainSet.npy")

    rows, cols = np.where(A != 0)
    Ind = np.zeros((rows.size, 2), dtype=np.int)
    for i in range(rows.size):
        Ind[i] = np.array([rows[i], cols[i]])
    return A, Ind


def RidgePostProcessing():
    # Kernel Ridge regression (gaussian kernel)
    A, Ind = initialization()

    final = np.load("results/RSVD.npy")

    for row, col in Ind:
        final[row, col] = A[row, col]
    factor = 5
    U, s, ZT = np.linalg.svd(final, full_matrices=True)
    D = np.diag(s[:factor])
    U = np.matmul(U[:, :factor], np.sqrt(D))
    ZT = np.matmul(np.sqrt(D), ZT[:factor, :])

    lambda_3 = 0.7

    postFinal = np.zeros((A.shape[0], A.shape[1]))

    for user in range(len(final)):
        if user % 100 == 0:
            print(user)
        rated_movies_indices = Ind[Ind[:, 0] == user, 1]  # vector of indices of movies rated by user user

        y = A[user, rated_movies_indices]  # vector of ratings to the rated movies by user user

        vj = [ZT.T[mov, :] for mov in rated_movies_indices]  # shape vj: num_rated_movies * 11

        norm = np.linalg.norm(vj, axis=1)

        vi = [ZT.T[mov, :] for mov in np.arange(0, 1000)]  # shape (1000-num_rated_movies) * 11

        norm_vi = np.linalg.norm(vi, axis=1)  # vector of shape (1000-num_rated_movies)

        Xi = np.zeros((len(vi), len(vi[0])))

        for i in range(len(vi)):
            Xi[i] = vi[i] / norm_vi[i]  # shape (1000-num_rated_movies) * 11

        X = np.zeros((len(vj), len(vj[0])))

        for i in range(len(vj)):
            X[i] = vj[i] / norm[i]  # shape X: num_rated_movies * 11

        s = np.matmul(np.exp(2 * (np.subtract(np.matmul(Xi, X.T), 1))), np.matmul(
            np.linalg.inv(np.exp(2 * (np.subtract(np.matmul(X, X.T), 1))) + lambda_3 * np.identity(len(vj))), y.T))

        postFinal[user, :] = s

    postFinal = np.clip(postFinal, 1, 5)
    print(postFinal[0, 9])
    print(postFinal)
    now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    np.save("Ridge" + now + "", postFinal)
    writeFile(postFinal)


num_users = 10000
num_movies = 1000
RidgePostProcessing()