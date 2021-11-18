import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

import function as fuc

# np.random.seed(0)


def Sparse_NMF(X, r, lamb=0.1, maxit=500):
    if not ((r < X.shape[0]) | (r < X.shape[1])):
        raise ValueError('Erro: Valor de r')
    H = np.random.rand(r, X.shape[1])
    D = np.random.rand(X.shape[0], r)
    Dnorm = D / np.sum(D ** 2, axis=0) ** (.5)
    for i in range(maxit):
        H = H * (np.dot(Dnorm.T, X)) / (np.dot(np.dot(Dnorm.T, Dnorm), H) + lamb)
        D = Dnorm * (np.dot(X, H.T) + Dnorm * (
            np.dot(np.ones((X.shape[0], X.shape[0])), np.dot(Dnorm, np.dot(H, H.T)) * Dnorm))) / (
                    np.dot(Dnorm, np.dot(H, H.T)) + Dnorm * (
                np.dot(np.ones((X.shape[0], X.shape[0])), np.dot(X, H.T) * Dnorm)))
        Dnorm = D / np.sum(D ** 2, axis=0) ** (.5)
    return D, H


iris = load_iris()
data = iris["data"]
lab = iris["target"]
classifer = KMeans(n_clusters=3)
D, H = Sparse_NMF(data.T, 3)
print(D,H)
print(fuc.acc(lab, classifer.fit_predict(H.T)), fuc.nmi(lab, classifer.fit_predict(H.T)))

