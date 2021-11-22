import function as fuc
from function import acc
from function import nmi
import SGNMF
from SGNMF import sp_clustering
from SGNMF import SGNMF_inv_Gaussian
from SGNMF import SGNMF_inv_Laplacian
from SGNMF import SGNMF_comp_inv
from SGNMF import SGNMF_hyper_tan
from SGNMF import SGNMF_Symmetric_CT
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise
from advanced_agolorithm import cfsfdp
from PIL import Image
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


# 计算距离矩阵
def compute_squared_EDM(X):
    return squareform(pdist(X, metric='euclidean'))


a = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])

dataset = [fuc.data_yale, fuc.data_USPS, fuc.data_umist, fuc.data_libras, fuc.data_JAFFE]
print(dataset[2].shape)
lab = [fuc.lab_yale, fuc.lab_USPS, fuc.lab_umist, fuc.lab_libras, fuc.lab_JAFFE]
classfiers1 = [fuc.KMeans(n_clusters=15, init="random"), fuc.KMeans(n_clusters=10, init="random"),
               fuc.KMeans(n_clusters=20, init="random"),
               fuc.KMeans(n_clusters=15, init="random"), fuc.KMeans(n_clusters=10, init="random")]
classfiers2 = [fuc.KMeans(n_clusters=15, init="k-means++"), fuc.KMeans(n_clusters=10, init="k-means++"),
               fuc.KMeans(n_clusters=20, init="k-means++"),
               fuc.KMeans(n_clusters=15, init="k-means++"), fuc.KMeans(n_clusters=10, init="k-means++")]
classfiers = [classfiers1, classfiers2]
r = [15, 10, 20, 15, 10]
# 调用cfsfdp算法
'''lab_cfsfdp = cfsfdp(dataset[3].T, r=0.65, classnum=r[3])
print(acc(lab[3], lab_cfsfdp), nmi(lab[3], lab_cfsfdp))'''
parameter = [[0.1, 2 * 30, 6, 0.01], [0.3, 2 * 0.0001, 3, 0.01], [30, 2 * 0.0003, 4, 0.001], [50, 2 * 2, 4, 2],
             [30, 2 * 0.1, 6, 10]]

title = ["yale", "USPS", "umist", "libras", "JAFFE"]
sigma = [29 / 5, 88 / 5, 111 / 5, 21 / 5, 14 / 5]

beta = [1e-3, 4e-3, 7e-3, 1e-2, 4e-3, 7e-2, 1e-1, 4e-1, 7e-1, 1, 4, 7, 10,40, 70, 100]#
cgma = [1e-4, 4e-4, 7e-4, 1e-3, 4e-3, 7e-3, 1e-2, 4e-3, 7e-2, 1e-1, 4e-1, 7e-1, 1, 4, 7, 10]

print(parameter[0][0],parameter[0][1])


for i in [ 2,4]:
    acc_all = np.array([])
    nmi_all = np.array([])
    for beta_1 in beta:
        acc_arr = np.array([])
        nmi_arr = np.array([])
        for j in range(10):
            U, V, err = SGNMF_inv_Gaussian(dataset[i], r=r[i], k=100, lamb=parameter[i][0], p=parameter[i][2],
                                      sigma=sigma[i],
                                      beta=beta_1,
                                      cgma=parameter[i][3], linmatrix=None, init="nndsvdar")
            lab_pre = classfiers2[i].fit_predict(V)
            a = acc(lab[i], lab_pre)
            b = nmi(lab[i], lab_pre)
            acc_arr = np.append(acc_arr, a)
            nmi_arr = np.append(nmi_arr, b)
            U, V, err = SGNMF_inv_Laplacian(dataset[i], r=r[i], k=100, lamb=parameter[i][0], p=parameter[i][2],
                                       sigma=sigma[i],
                                       beta=beta_1,
                                       cgma=parameter[i][3], linmatrix=None, init="nndsvdar")
            lab_pre = classfiers2[i].fit_predict(V)
            a = acc(lab[i], lab_pre)
            b = nmi(lab[i], lab_pre)
            acc_arr = np.append(acc_arr, a)
            nmi_arr = np.append(nmi_arr, b)
            U, V, err = SGNMF_comp_inv(dataset[i], r=r[i], k=100, lamb=parameter[i][0], p=parameter[i][2],
                                  sigma=sigma[i],
                                  beta=beta_1,
                                  cgma=parameter[i][3], linmatrix=None, init="nndsvdar")
            lab_pre = classfiers2[i].fit_predict(V)
            a = acc(lab[i], lab_pre)
            b = nmi(lab[i], lab_pre)
            acc_arr = np.append(acc_arr, a)
            nmi_arr = np.append(nmi_arr, b)
            U, V, err = SGNMF_Symmetric_CT(dataset[i], r=r[i], k=100, lamb=parameter[i][0], p=parameter[i][2],
                                      sigma=sigma[i],
                                      beta=beta_1,
                                      cgma=parameter[i][3], linmatrix=None, init="nndsvdar")
            lab_pre = classfiers2[i].fit_predict(V)
            a = acc(lab[i], lab_pre)
            b = nmi(lab[i], lab_pre)
            acc_arr = np.append(acc_arr, a)
            nmi_arr = np.append(nmi_arr, b)
            U, V, err= SGNMF_hyper_tan(dataset[i], r=r[i], k=100, lamb=parameter[i][0], p=parameter[i][2],
                                   sigma=sigma[i],
                                   beta=beta_1,
                                   cgma=parameter[i][3], linmatrix=None, init="nndsvdar")
            lab_pre = classfiers2[i].fit_predict(V)
            a = acc(lab[i], lab_pre)
            b = nmi(lab[i], lab_pre)
            acc_arr = np.append(acc_arr, a)
            nmi_arr = np.append(nmi_arr, b)
            print(a, b)
        print(acc_arr.shape)
        acc_mean = np.sum(acc_arr.reshape(10, 5), axis=0) / 10
        print(acc_mean.shape)

        nmi_mean = np.sum(nmi_arr.reshape(10, 5), axis=0) / 10
        print(acc_mean, nmi_mean)
        acc_all = np.append(acc_all, acc_mean)
        nmi_all = np.append(nmi_all, nmi_mean)
        print(acc_all.shape)
    np.save("parameter-setting\\accbeta" + title[i] + ".npy", acc_all.reshape(16, 5))
    np.save("parameter-setting\\nmibeta" + title[i] + ".npy", nmi_all.reshape(16, 5))


for i in [2,4]:
    acc_all = np.array([])
    nmi_all = np.array([])
    for cgma_1 in cgma:
        acc_arr = np.array([])
        nmi_arr = np.array([])
        for j in range(10):
            U, V, err = SGNMF_inv_Gaussian(dataset[i], r=r[i], k=100, lamb=parameter[i][0], p=parameter[i][2],
                                      sigma=sigma[i],
                                      beta=parameter[i][1],
                                      cgma=cgma_1, linmatrix=None, init="nndsvdar")
            lab_pre = classfiers2[i].fit_predict(V)
            a = acc(lab[i], lab_pre)
            b = nmi(lab[i], lab_pre)
            acc_arr = np.append(acc_arr, a)
            nmi_arr = np.append(nmi_arr, b)
            U, V, err = SGNMF_inv_Laplacian(dataset[i], r=r[i], k=100, lamb=parameter[i][0], p=parameter[i][2],
                                       sigma=sigma[i],
                                       beta=parameter[i][1],
                                       cgma=cgma_1, linmatrix=None, init="nndsvdar")
            lab_pre = classfiers2[i].fit_predict(V)
            a = acc(lab[i], lab_pre)
            b = nmi(lab[i], lab_pre)
            acc_arr = np.append(acc_arr, a)
            nmi_arr = np.append(nmi_arr, b)
            U, V, err = SGNMF_comp_inv(dataset[i], r=r[i], k=100, lamb=parameter[i][0], p=parameter[i][2],
                                  sigma=sigma[i],
                                  beta=parameter[i][1],
                                  cgma=cgma_1, linmatrix=None, init="nndsvdar")
            lab_pre = classfiers2[i].fit_predict(V)
            a = acc(lab[i], lab_pre)
            b = nmi(lab[i], lab_pre)
            acc_arr = np.append(acc_arr, a)
            nmi_arr = np.append(nmi_arr, b)
            U, V, err = SGNMF_Symmetric_CT(dataset[i], r=r[i], k=100, lamb=parameter[i][0], p=parameter[i][2],
                                      sigma=sigma[i],
                                      beta=parameter[i][1],
                                      cgma=cgma_1, linmatrix=None, init="nndsvdar")
            lab_pre = classfiers2[i].fit_predict(V)
            a = acc(lab[i], lab_pre)
            b = nmi(lab[i], lab_pre)
            acc_arr = np.append(acc_arr, a)
            nmi_arr = np.append(nmi_arr, b)
            U, V, err= SGNMF_hyper_tan(dataset[i], r=r[i], k=100, lamb=parameter[i][0], p=parameter[i][2],
                                   sigma=sigma[i],
                                   beta=parameter[i][1],
                                   cgma=cgma_1, linmatrix=None, init="nndsvdar")
            lab_pre = classfiers2[i].fit_predict(V)
            a = acc(lab[i], lab_pre)
            b = nmi(lab[i], lab_pre)
            acc_arr = np.append(acc_arr, a)
            nmi_arr = np.append(nmi_arr, b)
            print(a, b)
        print(acc_arr.shape)
        acc_mean = np.sum(acc_arr.reshape(10, 5), axis=0) / 10
        print(acc_mean.shape)
        nmi_mean = np.sum(nmi_arr.reshape(10, 5), axis=0) / 10
        acc_all = np.append(acc_all, acc_mean)
        nmi_all = np.append(nmi_all, nmi_mean)
        print(acc_all.shape)
    print(acc_all.shape)
    np.save("parameter-setting\\acccgma" + title[i] + ".npy", acc_all.reshape(16, 5))
    np.save("parameter-setting\\nmicgma" + title[i] + ".npy", nmi_all.reshape(16, 5))






for i in [0]:
    U, V, err_arr_5 = SGNMF_hyper_tan(dataset[i], r=r[i], k=100, lamb=5, p=6, sigma=sigma[i],
                                      beta=5,
                                      cgma=0.01, linmatrix=None, init="random")
    im = Image.fromarray(V.reshape(45, 55) * 2)
    im.convert("RGB").save("data" + title[i] + "hyT.png", quality=100)
    print("oookkkk")
    U, V, err_arr_1 = SGNMF_inv_Gaussian(dataset[i], r=r[i], k=100, lamb=5, p=4, sigma=sigma[i],
                                         beta=5,
                                         cgma=10, linmatrix=None, init="random")
    im = Image.fromarray(V.reshape(90, 60) * 255 / 2)
    im.convert("RGB").save("data" + title[i] + "inG.png", quality=100, format="png")
    U, V, err_arr_5 = SGNMF_hyper_tan(dataset[i], r=r[i], k=100, lamb=0, p=4, sigma=sigma[i],
                                      beta=0,
                                      cgma=0.01, linmatrix=None, init="random")
    im = Image.fromarray(V.reshape(90, 60) * 255 / 2)
    print("ok")
    im.convert("RGB").save("data" + title[i] + "NMF.png", quality=100)
    U, V, err_arr_5 = SGNMF_hyper_tan(dataset[i], r=r[i], k=100, lamb=5, p=4, sigma=sigma[i],
                                      beta=0,
                                      cgma=0.01, linmatrix=None, init="random")
    im = Image.fromarray(V.reshape(90, 60) * 255 / 2)
    im.convert("RGB").save("data" + title[i] + "GNMF.png", quality=100)
    U, V, err_arr_2 = SGNMF_inv_Laplacian(dataset[i], r=r[i], k=100, lamb=5, p=4, sigma=sigma[i],
                                          beta=5,
                                          cgma=0.01, linmatrix=None, init="random")
    im = Image.fromarray(V.reshape(90, 60) * 255 / 2)
    im.convert("RGB").save("data" + title[i] + "inL.png")
    U, V, err_arr_3 = SGNMF_comp_inv(dataset[i], r=r[i], k=100, lamb=5, p=4, sigma=sigma[i],
                                     beta=5,
                                     cgma=0.01, linmatrix=None, init="random")
    im = Image.fromarray(V.reshape(90, 60) * 255 / 2)
    im.convert("RGB").save("data" + title[i] + "coI.png", quality=100)
    U, V, err_arr_4 = SGNMF_Symmetric_CT(dataset[i], r=r[i], k=100, lamb=5, p=4, sigma=sigma[i],
                                         beta=5,
                                         cgma=0.01, linmatrix=None, init="random")
    im = Image.fromarray(V.reshape(90, 60) * 255 / 2)
    im.convert("RGB").save("data" + title[i] + "syC.png", quality=100)

    err_arr = [err_arr_1, err_arr_2, err_arr_3, err_arr_4, err_arr_5]
    SGNMF.draw(err_arr, title[i])
    print("ok")


# 调用kmeans
def kmean_train(i):
    lab1 = classfiers2[i].fit_predict(dataset[i].T)
    return acc(lab[i], lab1), nmi(lab[i], lab1)


usps_all = np.load("../SGNMF_more_function/dataset/usps_all.npy")
usps_lab_all = np.load("../SGNMF_more_function/dataset/usps_lab_all.npy")
'''for i in range(20):
    print(sp_clustering(usps_all.T,usps_lab_all,r = r[1],p =9,sigma=50/5))
    lab_all = 0
    print(nmi(usps_lab_all,lab_all),acc(usps_lab_all,lab_all))'''

# 谱聚类,kmeans调差
'''for m in [2]:
    acc_arr = np.array([])
    nmi_arr = np.array([])
    for j in range(20):
        a, b = sp_clustering(dataset[m].T, lab[m], r=r[m], p=5, sigma=111 / 5)
        acc_arr = np.append(acc_arr, a)
        nmi_arr = np.append(nmi_arr, b)
        print(a, b)
    print(np.mean(acc_arr), np.std(acc_arr))
    print(np.mean(nmi_arr), np.std(nmi_arr))'''
acc_mean_inG = []
acc_std_inG = []
nmi_mean_inG = []
nmi_std_inG = []
p = []
beta = []
lamb = []
# 训练代码
for cgma_1 in [0.01, 0.0001, 0.005, 0.009, 0.01, 0.05, 0.09, 0.1, 0.5, 0.9, 1, 5, 9, 10]:
    for i in [0]:
        for beta_1 in [30, 100, 0.005, 6e-5, 0.00008, 0.0001, 0.0003, 0.0007]:
            for lamb_1 in [0.1, 1]:

                # adj_matrix = np.load("similarity_matrix\\dateset" + str(i) + "p=" + str(p_1) + ".npy")

                for p_1 in [6, 4, 5, 8]:
                    for h in [0]:
                        print(h, i, lamb_1, p_1, beta_1)
                        acc_arr = np.array([])
                        nmi_arr = np.array([])
                        for j in range(20):
                            U, V = SGNMF_hyper_tan(dataset[i], r=r[i] * 2, k=100, lamb=lamb_1, p=p_1,
                                                   sigma=sigma[i] * 2 - h,
                                                   beta=beta_1,
                                                   cgma=cgma_1, linmatrix=None, init="nndsvda")
                            lab_pre = classfiers2[i].fit_predict(V)
                            a = acc(lab[i], lab_pre)
                            b = nmi(lab[i], lab_pre)
                            acc_arr = np.append(acc_arr, a)
                            nmi_arr = np.append(nmi_arr, b)
                            print(a, b)
                        acc_mean_inG.append(np.mean(acc_arr))
                        acc_std_inG.append(np.std(acc_arr))
                        nmi_mean_inG.append(np.mean(nmi_arr))
                        nmi_std_inG.append(np.std(nmi_arr))
                        print(np.mean(acc_arr), np.std(acc_arr))
                        print(np.mean(nmi_arr), np.std(nmi_arr))
                        p.append(p_1)
                        beta.append(beta_1)
                        lamb.append(lamb_1)
        df = pd.DataFrame({"p": p, "beta": beta, "lamb": lamb, "acc_mean": acc_mean_inG, "acc_std": acc_std_inG,
                           "nmi_mean": nmi_mean_inG, "nmi_std": nmi_std_inG})
        df.to_csv("cgma=" + str(cgma_1) + "_inG_dataset" + str(i) + ".csv")

# 调参代码
'''lamb = [0.1, 0.5, 1, 5, 10, 50, 100]
p = [3, 6, 9]
beta = [0.1, 0.5, 1, 5, 10, 50]
cgma = [0.1]
acc_inG = []
acc_inL = []
acc_coI = []
acc_sy_c = []
acc_hy_T = []
nmi_inG = []
nmi_inL = []
nmi_coI = []
nmi_sy_C = []
nmi_hy_T = []
# 看两个样本之间的距离
for i in range(5):
    print(np.mean(np.sum((dataset[i] - dataset[i][0]) ** 2, axis=1)))
lamb_csv = []
sigma_csv = []
p_csv = []
beta_csv = []

# 计算每个数据集的相似度矩阵
for i in range(5):
    for p_1 in p:
        a = SGNMF.calEuclidDistanceMatrix(dataset[i].T)
        b = SGNMF.myKNN(a, k=p_1, sigma=sigma[i])
        np.save("similarity_matrix\\dateset"+str(i)+"p="+str(p_1)+".npy", b)

for i in [1, 2, 3, 4]:
    for lamb_1 in lamb:
        for p_1 in p:
            adj_matrix = np.load("similarity_matrix\\dateset" + str(i) + "p=" + str(p_1) + ".npy")
            for beta_1 in beta:
                print(i, lamb_1, p_1, beta_1)
                U, V = SGNMF_inv_Gaussian(dataset[i], r=r[i], k=100, lamb=lamb_1, p=p_1, sigma=sigma[i], beta=beta_1,
                                          cgma=0.05, linmatrix=adj_matrix)
                lab_pre = classfiers2[i].fit_predict(V)
                acc_inG.append(acc(lab[i], lab_pre))
                nmi_inG.append(nmi(lab[i], lab_pre))
                print(acc(lab[i], lab_pre))
                U, V = SGNMF_inv_Laplacian(dataset[i], r=r[i], k=100, lamb=lamb_1, p=p_1, sigma=sigma[i], beta=beta_1,
                                           cgma=0.05, linmatrix=adj_matrix)
                lab_pre = classfiers2[i].fit_predict(V)
                acc_inL.append(acc(lab[i], lab_pre))
                nmi_inL.append(nmi(lab[i], lab_pre))
                print(acc(lab[i], lab_pre))
                U, V = SGNMF_comp_inv(dataset[i], r=r[i], k=100, lamb=lamb_1, p=p_1, sigma=sigma[i], beta=beta_1,
                                      cgma=0.05, linmatrix=adj_matrix)
                lab_pre = classfiers2[i].fit_predict(V)
                acc_coI.append(acc(lab[i], lab_pre))
                nmi_coI.append(nmi(lab[i], lab_pre))
                print(acc(lab[i], lab_pre))

                U, V = SGNMF_Symmetric_CT(dataset[i], r=r[i], k=100, lamb=lamb_1, p=p_1, sigma=sigma[i], beta=beta_1,
                                          cgma=0.05, linmatrix=adj_matrix)
                lab_pre = classfiers2[i].fit_predict(V)
                acc_sy_c.append(acc(lab[i], lab_pre))
                nmi_sy_C.append(nmi(lab[i], lab_pre))
                print(acc(lab[i], lab_pre))

                U, V = SGNMF_hyper_tan(dataset[i], r=r[i], k=100, lamb=lamb_1, p=p_1, sigma=sigma[i], beta=beta_1,
                                       cgma=0.05, linmatrix=adj_matrix)
                lab_pre = classfiers2[i].fit_predict(V)
                acc_hy_T.append(acc(lab[i], lab_pre))
                nmi_hy_T.append(nmi(lab[i], lab_pre))
                print(acc(lab[i], lab_pre))
                lamb_csv.append(lamb_1)
                p_csv.append(p_1)
                beta_csv.append(beta_1)
                # , "acc_inL": acc_inL,"nmi_inL": nmi_inL,"acc_hyT": acc_hy_T, "nmi_hyT": nmi_hy_T,"acc_inG": acc_inG, "nmi_inG": nmi_inG, "acc_coI": acc_coI, "nmi_coI": nmi_coI, "acc_sy_C": acc_sy_c, "nmi_sy_C": nmi_sy_C
    print(len(beta_csv), len(p_csv), len(acc_hy_T))
    df = pd.DataFrame(
        {"lamda": lamb_csv, "beta": beta_csv, "p": p_csv, "acc_sy_C": acc_sy_c, "nmi_sy_C": nmi_sy_C})
    # df.to_csv("数据集" + str(i) + "调参syC.csv")
    acc_inG = []
    acc_inL = []
    acc_coI = []
    acc_sy_c = []
    acc_hy_T = []
    nmi_inG = []
    nmi_inL = []
    nmi_coI = []
    nmi_sy_C = []
    nmi_hy_T = []
    lamb_csv = []
    sigma_csv = []
    p_csv = []
    beta_csv = []'''

'''for j in range(2,10):
    j1 = 2*j
    classifer = KMeans(n_clusters=j1)
    result_GGNMF = np.array([])
    result_NMF = np.array([])
    result_kmeans = np.array([])
    result_spcluter = np.array([])
    for j in range(20):
        b = np.array([], dtype=np.int)
        a = np.arange(1, 21)
        np.random.shuffle(a)
        for i in range(j1):
            print(a[i], end=" ")
            temp = np.arange(72 * (a[i] - 1), 72 * a[i])
            b = np.append(b, temp)
        print()
        data1 = data[:, b]
        lab1 = lab[b]
        accuracy = NMF_train(data1,lab1,j1,200)
        result_spcluter = np.append(result_spcluter, accuracy)
        print(accuracy)
    print("averange:", np.mean(result_spcluter))
    print("standard var:", np.std(result_spcluter))'''
