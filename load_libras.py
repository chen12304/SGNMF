import numpy as np
import pandas as pd
import function as fuc

df = pd.read_csv("orignal dataset\\libras\\libras_data360 90.csv")

data=df.values
print(data)
print(data.shape)
data_ = data[:,0:90]
np.save("dataset\\libras.npy",data_.T)
print(data[:,0:90].T.shape,data[:,90].shape)

lab = np.array(data[:,90],dtype = np.int)

np.save("dataset\\libras_lab.npy",lab)
cls = fuc.KMeans(n_clusters=15)
print(
    fuc.acc(lab, cls.fit_predict((data_)))
)
