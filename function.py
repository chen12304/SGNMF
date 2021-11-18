from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from advanced_agolorithm import *
import scipy.io as scio
import datasets
'''print(datasets.list_datasets())
print(datasets.load_dataset("mnist"))'''
from sklearn.decomposition import nmf


def acc(y_true, y_pred):
    '''    Calculate clustering accuracy. Require scikit-learn installed
     Arguments        y: true labels, numpy.array with shape `(n_samples,)` y_pred: predicted labels, numpy.array with shape `(n_samples,)`
     Return        accuracy, in [0,1]    '''
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


wine = load_wine()
data_wine = wine["data"]
lab_wine = wine["target"]

iris = load_iris()
data_iris = iris["data"]
lab_iris = iris["target"]

data_JAFFE = np.load("dataset\\JAFFE.npy")
lab_JAFFE = np.load("dataset\\JAFFE_lab.npy")

data_libras = np.load("dataset\\libras.npy")
# data_libras = data_libras/(np.max(data_libras,axis=0)-np.min(data_libras,axis=0))
lab_libras = np.load("dataset\\libras_lab.npy")
# print(data_libras,lab_libras)
# print(data_libras.shape,lab_libras.shape)

data_umist = np.load("dataset\\umist.npy")/255
lab_umist = np.load("dataset\\umist_lab.npy")
'''for i in range(data_umist.shape[1]):
    print(np.sum((data_umist[:,i]-data_umist[:,i+1])**2))
    plt.imshow(data_umist[:,i].reshape(28,23)*255)
    plt.show()'''
'''print(data_umist.shape,lab_umist.shape)
classifer = KMeans(n_clusters=20,init="random")
for i in range(200):
    print(acc(lab_umist,classifer.fit_predict(data_umist.T)))'''

data_USPS = np.load("dataset\\usps.npy")
lab_USPS = np.load("dataset\\usps_lab.npy")

data_yale = np.load("dataset\\yale.npy")
lab_yale = np.load("dataset\\yale_lab.npy")



# print(data_yale.shape,lab_yale.shape)

# scio.savemat("dataset\\umist.mat", {"X": np.array(data_umist.T,dtype=np.float), "y": lab_umist.reshape(-1,1)})
