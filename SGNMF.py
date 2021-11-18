# coding: utf-8
import sys
import codecs

sys.stdout = codecs.getwriter('utf8')(sys.stdout.detach())

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.metrics.pairwise import cosine_similarity
import function

from sklearn.decomposition._nmf import _initialize_nmf





def get_col_20(file_path):
    im_array = np.array([])
    lab = np.array([], dtype=np.int)
    for file in os.listdir(file_path):
        im = Image.open(file_path + '\\' + file)
        im = np.array(im)
        im_array = np.append(im_array, im / 255)  # 同理归一化
        lab = np.append(lab, int(eval(file[3:file.find('_')])))
    return im_array.reshape(1440, 32 * 32).T, lab


def eachFile(filepath):
    im_array = np.array([])
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if os.path.splitext(file)[1] == '.pgm':
                # print(os.path.join(root, file))
                im = Image.open(os.path.join(root, file))
                # im.show()
                # print(im.size)
                im_array = np.append(im_array, np.array(im) / 255)  # 数据归一化在加载图片是需要把数据归一化否则带入高斯核函数很难调参，并且出现过拟合和欠拟合的情况
    return im_array.reshape(400, 92 * 112).T


def load_lab_orl(lab_orl, num):
    if num > 0:
        return load_lab_orl(np.append(lab_orl, num * np.ones(10, dtype=int)), num - 1)
    else:
        return lab_orl


# 训练相似矩阵W，使用余弦相似度
def trainW(v):
    similarMatrix = cosine_similarity(v.T)
    # similarMatrix = pairwise_distances(v,metric="cosine")
    m = np.shape(similarMatrix)[0]
    print(m)
    for i in range(m):
        for j in range(m):
            if j == i:
                similarMatrix[i][j] = 0
    return similarMatrix


def euclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1 - x2) ** 2)
    # print(res)
    if sqrt_flag:
        res = np.sqrt(res)
    return res


def calEuclidDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
            # print(S[i][j])
    return S


def myKNN(S, k, sigma=1.0):
    N = len(S)
    A = np.zeros((N, N))
    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x: x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k + 1)]  # xi's k nearest neighbours

        for j in neighbours_id:  # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j] / 2 / sigma / sigma)
            A[j][i] = A[i][j]  # mutually
    return A


def calLaplacianMatrix(adjacentMatrix):
    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)


def sp_clustering(data, lab_sp, r, p, sigma):
    '''谱聚类：分别为数据，标签，类数，邻近数，以及权重参数'''
    sp_classifer = SpectralClustering(n_clusters=r, affinity="nearest_neighbors", n_neighbors=p, gamma=sigma)
    # print(lab_sp.shape, sp_classifer.fit_predict(data).shape)
    lab = sp_classifer.fit_predict(data)
    return function.acc(lab_sp, lab),function.nmi(lab_sp,lab)


def SGNMF_inv_Gaussian(V, r, k, lamb=10, p=5, sigma=1, beta=1, cgma=1.0, linmatrix=None,init = "random"):
    """此函数以倒高斯函数作为稀疏项
    V是原始数据，一列一个数据
    r为类别数
    k为循环次数
    lamb为GNMF前的值
    p为邻近数
    sigma为高斯核函数中的值
    beta为倒高斯函数系数
    cgma为倒高斯函数里指数上的参数"""
    m, n = np.shape(V)
    # 先随机给定一个W、H，保证矩阵的大小
    H,W = _initialize_nmf(V.T,n_components=r,init=init)
    W = W.T



    D = []
    if linmatrix is not None:
        linMatrix = linmatrix
    else:
        trainV = V
        # similarMatrix = trainW(trainV) #使用的是余弦相似度
        similarMatrix = calEuclidDistanceMatrix(trainV.T)
        # similarMatrix = np.load("coil-similarMatrix.npy")
        # print("similarM.shape:", similarMatrix.shape)
        linMatrix = myKNN(similarMatrix, p, sigma)  # 使用01或者高斯核函数赋权重

    # print("最近邻矩阵：", linMatrix)
    # print("最近邻矩阵的规格：", linMatrix.shape)
    D = np.diag(np.sum(linMatrix, axis=0))
    # print('degree matrix:', D)
    # print("度矩阵的规格：", D.shape)

    # K为迭代次数

    err_arr = np.array([])
    for x in range(k):

        # 权值更新
        a = np.dot(V.T, W) + lamb * np.dot(linMatrix, H)
        b = np.dot(np.dot(H, W.T), W) + lamb * np.dot(D, H) + beta * np.exp(
            -H ** 2 / cgma / cgma) / cgma / cgma
        for i_1 in range(n):
            for j_1 in range(r):
                if b[i_1, j_1] != 0:
                    H[i_1, j_1] = H[i_1, j_1] * (a[i_1, j_1] / b[i_1, j_1])
        c = np.dot(V, H)
        d = np.dot(np.dot(W, H.T), H)
        for i_2 in range(m):
            for j_2 in range(r):
                if d[i_2, j_2] != 0:
                    W[i_2, j_2] = W[i_2, j_2] * (c[i_2, j_2] / d[i_2, j_2])
        e = (np.sum(W**2, axis=0))**0.5
        M = np.diag(1 / e)
        W = np.dot(W, M)
        M = np.diag(e)
        H = np.dot(H, M)
        '''
        e = (np.sum(H ** 2, axis=0)) ** 0.5
        M = np.diag(1 / e)
        H = np.dot(H, M)
        M = np.diag(e)
        W = np.dot(W, M)'''
        '''if x%10==0:
            SF = np.sum(H>1e-10)
            sprasity = np.append(sprasity, SF)
            print(SF)'''
        err_arr = np.append(err_arr, np.sum((V - np.dot(W, H.T)) ** 2) + lamb * np.sum(
            np.diag(np.dot(H.T, np.dot(D - linMatrix, H)))) + beta * np.sum(1 - np.exp(-H ** 2 / cgma / cgma)))
        # print(np.sum((V - np.dot(W, H.T)) ** 2) + lamb * np.sum(
        #   np.diag(np.dot(H.T, np.dot(D - linMatrix, H)))) + beta * np.sum(1 - np.exp(-H ** 2 / cgma / cgma / 2)))
    # np.save("sprasity_get\\coil\\coil-20-10.npy", sprasity)
    return W, H


def SGNMF_inv_Laplacian(V, r, k, lamb=10, p=10, sigma=50, beta=1, cgma=100.0, linmatrix=None,init = "random"):
    """此函数为倒拉普拉斯函数作为稀疏项
    V是原始数据，一列一个数据
    r为类别数
    k为循环次数
    lamb为GNMF前的值
    p为邻近数
    sigma为高斯核函数中的值
    beta为倒拉普拉斯函数系数
    cgma为倒拉普拉斯函数里指数上的参数"""
    m, n = np.shape(V)
    # 先随机给定一个W、H，保证矩阵的大小
    H, W = _initialize_nmf(V.T, n_components=r, init=init)
    W = W.T

    D = []
    if linmatrix is not None:
        linMatrix = linmatrix
    else:
        trainV = V
        # similarMatrix = trainW(trainV)
        similarMatrix = calEuclidDistanceMatrix(trainV.T)
        # similarMatrix = np.load("coil-similarMatrix.npy")
        # print("similarM.shape:", similarMatrix.shape)
        linMatrix = myKNN(similarMatrix, p, sigma)

    # print("最近邻矩阵：", linMatrix)
    # print("最近邻矩阵的规格：", linMatrix.shape)
    D = np.diag(np.sum(linMatrix, axis=0))
    # print('degree matrix:', D)
    # print("度矩阵的规格：", D.shape)

    # K为迭代次数

    err_arr = np.array([])
    for x in range(k):

        # 权值更新
        a = np.dot(V.T, W) + lamb * np.dot(linMatrix, H)
        b = np.dot(np.dot(H, W.T), W) + lamb * np.dot(D, H) + beta * np.exp(-H / cgma) / cgma
        for i_1 in range(n):
            for j_1 in range(r):
                if b[i_1, j_1] != 0:
                    H[i_1, j_1] = H[i_1, j_1] * (a[i_1, j_1] / b[i_1, j_1])
        c = np.dot(V, H)
        d = np.dot(np.dot(W, H.T), H)
        for i_2 in range(m):
            for j_2 in range(r):
                if d[i_2, j_2] != 0:
                    W[i_2, j_2] = W[i_2, j_2] * (c[i_2, j_2] / d[i_2, j_2])
        e = (np.sum(W ** 2, axis=0)) ** 0.5
        M = np.diag(1 / e)
        W = np.dot(W, M)
        M = np.diag(e)
        H = np.dot(H, M)
        '''
        e = (np.sum(H ** 2, axis=0)) ** 0.5
        M = np.diag(1 / e)
        H = np.dot(H, M)
        M = np.diag(e)
        W = np.dot(W, M)'''
        '''if x%10==0:
            SF = np.sum(H>1e-10)
            sprasity = np.append(sprasity, SF)
            print(SF)'''
        err_arr = np.append(err_arr, np.sum((V - np.dot(W, H.T)) ** 2) + lamb * np.sum(
            np.diag(np.dot(H.T, np.dot(D - linMatrix, H)))) + beta * np.sum(1 - np.exp(-H / cgma)))
        # print(np.sum((V - np.dot(W, H.T)) ** 2) + lamb * np.sum(
        #   np.diag(np.dot(H.T, np.dot(D - linMatrix, H)))) + beta * np.sum(1 - np.exp(-H ** 2 / cgma / cgma / 2)))
    # np.save("sprasity_get\\coil\\coil-20-10.npy", sprasity)
    return W, H


def SGNMF_hyper_tan(V, r, k, lamb=10, p=10, sigma=50, beta=1, cgma=100.0, linmatrix=None,init = "random"):
    """此函数为hyperbolic tangent function作为稀疏项
    V是原始数据，一列一个数据
    r为类别数
    k为循环次数
    lamb为GNMF前的值
    p为邻近数
    sigma为高斯核函数中的值
    beta为hyperbolic tangent function系数
    cgma为hyperbolic tangent function的参数"""
    m, n = np.shape(V)
    # 先随机给定一个W、H，保证矩阵的大小
    H, W = _initialize_nmf(V.T, n_components=r, init=init)
    W = W.T
    '''W = np.array(np.random.random((m, r)))
    H = np.array(np.random.random((n, r)))'''

    D = []
    if linmatrix is not None:
        linMatrix = linmatrix
    else:
        trainV = V
        # similarMatrix = trainW(trainV)
        similarMatrix = calEuclidDistanceMatrix(trainV.T)
        # similarMatrix = np.load("coil-similarMatrix.npy")
        # print("similarM.shape:", similarMatrix.shape)
        linMatrix = myKNN(similarMatrix, p, sigma)

    # print("最近邻矩阵：", linMatrix)
    # print("最近邻矩阵的规格：", linMatrix.shape)
    D = np.diag(np.sum(linMatrix, axis=0))
    # print('degree matrix:', D)
    # print("度矩阵的规格：", D.shape)

    # K为迭代次数

    err_arr = np.array([])
    for x in range(k):

        # 权值更新
        a = np.dot(V.T, W) + lamb * np.dot(linMatrix, H)
        b = np.dot(np.dot(H, W.T), W) + lamb * np.dot(D, H) + beta /(H +cgma)**2
        for i_1 in range(n):
            for j_1 in range(r):
                if b[i_1, j_1] != 0:
                    H[i_1, j_1] = H[i_1, j_1] * (a[i_1, j_1] / b[i_1, j_1])
        c = np.dot(V, H)
        d = np.dot(np.dot(W, H.T), H)
        for i_2 in range(m):
            for j_2 in range(r):
                if d[i_2, j_2] != 0:
                    W[i_2, j_2] = W[i_2, j_2] * (c[i_2, j_2] / d[i_2, j_2])
        e = (np.sum(W ** 2, axis=0)) ** 0.5
        M = np.diag(1 / e)
        W = np.dot(W, M)
        M = np.diag(e)
        H = np.dot(H, M)
        '''
        e = (np.sum(H ** 2, axis=0)) ** 0.5
        M = np.diag(1 / e)
        H = np.dot(H, M)
        M = np.diag(e)
        W = np.dot(W, M)'''
        '''if x%10==0:
            SF = np.sum(H>1e-10)
            sprasity = np.append(sprasity, SF)
            print(SF)'''
        '''err_arr = np.append(err_arr, np.sum((V - np.dot(W, H.T)) ** 2) + lamb * np.sum(
            np.diag(np.dot(H.T, np.dot(D - linMatrix, H)))) + beta * np.sum(
            (np.exp(2 * H * H / cgma / cgma) - 1) / (np.exp(2 * H * H / cgma / cgma) + 1)))'''
        # print(np.sum((V - np.dot(W, H.T)) ** 2) + lamb * np.sum(
        #   np.diag(np.dot(H.T, np.dot(D - linMatrix, H)))) + beta * np.sum(1 - np.exp(-H ** 2 / cgma / cgma / 2)))
    # np.save("sprasity_get\\coil\\coil-20-10.npy", sprasity)
    return W, H


def SGNMF_Symmetric_CT(V, r, k, lamb=10, p=10, sigma=50, beta=1, cgma=100.0, linmatrix=None,init = "random"):
    """此函数为Symmetric.CT function作为稀疏项
    V是原始数据，一列一个数据
    r为类别数
    k为循环次数
    lamb为GNMF前的值
    p为邻近数
    sigma为高斯核函数中的值
    beta为Symmetric.CT function系数
    cgma为Symmetric.CT function的参数"""
    m, n = np.shape(V)
    # 先随机给定一个W、H，保证矩阵的大小
    H, W = _initialize_nmf(V.T, n_components=r, init=init)
    W = W.T

    D = []
    if linmatrix is not None:
        linMatrix = linmatrix
    else:
        trainV = V
        # similarMatrix = trainW(trainV)
        similarMatrix = calEuclidDistanceMatrix(trainV.T)
        # similarMatrix = np.load("coil-similarMatrix.npy")
        # print("similarM.shape:", similarMatrix.shape)
        linMatrix = myKNN(similarMatrix, p, sigma)

    # print("最近邻矩阵：", linMatrix)
    # print("最近邻矩阵的规格：", linMatrix.shape)
    D = np.diag(np.sum(linMatrix, axis=0))
    # print('degree matrix:', D)
    # print("度矩阵的规格：", D.shape)

    # K为迭代次数

    err_arr = np.array([])
    for x in range(k):

        # 权值更新
        a = np.dot(V.T, W) + lamb * np.dot(linMatrix, H)
        y = (H * H / cgma / cgma)
        b = np.dot(np.dot(H, W.T), W) + lamb * np.dot(D, H) + beta * np.cos(np.arctan(y)) / (
                1 + y ** 2) * 2 * H / cgma / cgma
        for i_1 in range(n):
            for j_1 in range(r):
                if b[i_1, j_1] != 0:
                    H[i_1, j_1] = H[i_1, j_1] * (a[i_1, j_1] / b[i_1, j_1])
        c = np.dot(V, H)
        d = np.dot(np.dot(W, H.T), H)
        for i_2 in range(m):
            for j_2 in range(r):
                if d[i_2, j_2] != 0:
                    W[i_2, j_2] = W[i_2, j_2] * (c[i_2, j_2] / d[i_2, j_2])
        e = (np.sum(W ** 2, axis=0)) ** 0.5
        M = np.diag(1 / e)
        W = np.dot(W, M)
        M = np.diag(e)
        H = np.dot(H, M)
        '''
        e = (np.sum(H ** 2, axis=0)) ** 0.5
        M = np.diag(1 / e)
        H = np.dot(H, M)
        M = np.diag(e)
        W = np.dot(W, M)'''
        '''if x%10==0:
            SF = np.sum(H>1e-10)
            sprasity = np.append(sprasity, SF)
            print(SF)'''
        err_arr = np.append(err_arr, np.sum((V - np.dot(W, H.T)) ** 2) + lamb * np.sum(
            np.diag(np.dot(H.T, np.dot(D - linMatrix, H)))) + beta * np.sum(np.sin(np.arctan(H ** 2 / cgma ** 2))))
        # print(np.sum((V - np.dot(W, H.T)) ** 2) + lamb * np.sum(
        #   np.diag(np.dot(H.T, np.dot(D - linMatrix, H)))) + beta * np.sum(1 - np.exp(-H ** 2 / cgma / cgma / 2)))
    # np.save("sprasity_get\\coil\\coil-20-10.npy", sprasity)
    return W, H


def SGNMF_comp_inv(V, r, k, lamb=10, p=10, sigma=50, beta=1, cgma=100.0, linmatrix=None,init = "random"):
    """此函数为comp.inv function作为稀疏项
    V是原始数据，一列一个数据
    r为类别数
    k为循环次数
    lamb为GNMF前的值
    p为邻近数
    sigma为高斯核函数中的值
    beta为comp.inv function 系数
    cgma为comp.inv function的参数"""
    m, n = np.shape(V)
    # 先随机给定一个W、H，保证矩阵的大小
    H, W = _initialize_nmf(V.T, n_components=r, init=init,random_state=1)
    W = W.T

    D = []
    if linmatrix is not None:
        linMatrix = linmatrix
    else:
        trainV = V
        # similarMatrix = trainW(trainV)
        similarMatrix = calEuclidDistanceMatrix(trainV.T)
        # similarMatrix = np.load("coil-similarMatrix.npy")
        # print("similarM.shape:", similarMatrix.shape)
        linMatrix = myKNN(similarMatrix, p, sigma)

    # print("最近邻矩阵：", linMatrix)
    # print("最近邻矩阵的规格：", linMatrix.shape)
    D = np.diag(np.sum(linMatrix, axis=0))
    # print('degree matrix:', D)
    # print("度矩阵的规格：", D.shape)

    # K为迭代次数

    err_arr = np.array([])
    for x in range(k):

        # 权值更新
        a = np.dot(V.T, W) + lamb * np.dot(linMatrix, H)
        b = np.dot(np.dot(H, W.T), W) + lamb * np.dot(D, H) + cgma ** 2 * 2 * H / (H ** 2 + cgma ** 2) ** 2
        for i_1 in range(n):
            for j_1 in range(r):
                if b[i_1, j_1] != 0:
                    H[i_1, j_1] = H[i_1, j_1] * (a[i_1, j_1] / b[i_1, j_1])
        c = np.dot(V, H)
        d = np.dot(np.dot(W, H.T), H)
        for i_2 in range(m):
            for j_2 in range(r):
                if d[i_2, j_2] != 0:
                    W[i_2, j_2] = W[i_2, j_2] * (c[i_2, j_2] / d[i_2, j_2])
        e = (np.sum(W ** 2, axis=0)) ** 0.5
        M = np.diag(1 / e)
        W = np.dot(W, M)
        M = np.diag(e)
        H = np.dot(H, M)
        '''
        e = (np.sum(H ** 2, axis=0)) ** 0.5
        M = np.diag(1 / e)
        H = np.dot(H, M)
        M = np.diag(e)
        W = np.dot(W, M)'''
        '''if x%10==0:
            SF = np.sum(H>1e-10)
            sprasity = np.append(sprasity, SF)
            print(SF)'''
        err_arr = np.append(err_arr, np.sum((V - np.dot(W, H.T)) ** 2) + lamb * np.sum(
            np.diag(np.dot(H.T, np.dot(D - linMatrix, H)))) + beta * np.sum(H ** 2 / (H ** 2 + cgma ** 2)))
        # print(np.sum((V - np.dot(W, H.T)) ** 2) + lamb * np.sum(
        #   np.diag(np.dot(H.T, np.dot(D - linMatrix, H)))) + beta * np.sum(1 - np.exp(-H ** 2 / cgma / cgma / 2)))
    # np.save("sprasity_get\\coil\\coil-20-10.npy", sprasity)
    return W, H




'''classfier = KMeans(n_clusters=3)

U, V = SGNMF_inv_Laplacian(function.data_iris.T, 3, 100, lamb=10, p=5, sigma=1, beta=10, cgma=1)
print(function.acc(function.lab_iris, classfier.fit_predict(V)))'''

