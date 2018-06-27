import numpy as np
import pandas as pd
import datetime
import time
import IOUtils


def RidgePostProcessing():
    A, Ind = IOUtils.initialization()

    final = np.load("../../CIL_results/results/RSVD.npy")

    for row, col in Ind:
        final[row, col] = A[row, col]
    factor = 12
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

        vj = [ZT.T[mov, :] for mov in rated_movies_indices]  # shape vj: num_rated_movies * #factors

        norm = np.linalg.norm(vj, axis=1)

        vi = [ZT.T[mov, :] for mov in np.arange(0, 1000)]  # shape (1000-num_rated_movies) * #factors

        norm_vi = np.linalg.norm(vi, axis=1)  # vector of shape (1000-num_rated_movies)

        Xi = np.zeros((len(vi), len(vi[0])))

        for i in range(len(vi)):
            Xi[i] = vi[i] / norm_vi[i]  # shape (1000-num_rated_movies) * #factors

        X = np.zeros((len(vj), len(vj[0])))

        for i in range(len(vj)):
            X[i] = vj[i] / norm[i]  # shape X: num_rated_movies * #factors

        s = np.matmul(np.exp(2 * (np.subtract(np.matmul(Xi, X.T), 1))), np.matmul(
            np.linalg.inv(np.exp(2 * (np.subtract(np.matmul(X, X.T), 1))) + lambda_3 * np.identity(len(vj))), y.T))

        postFinal[user, :] = s

    postFinal = np.clip(postFinal, 1, 5)

    now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    np.save("Ridge" + now + "", postFinal)
    #IOUtils.writeFile(postFinal)


def get_rating(user, movie, U, ZT, b, b_u, b_m_weighted):
    return b_u[user] + b_m_weighted[movie] + U[user, :].dot(ZT[:, movie])


def getFullMatrix(A, U, ZT, b, b_u, b_m_weighted):
    fullMatrix = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            fullMatrix[i, j] = get_rating(i, j,U, ZT, b, b_u, b_m_weighted)
    return fullMatrix


def main():
    RidgePostProcessing()



if __name__ == '__main__':
    main()