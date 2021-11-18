import os
from PIL import Image
import numpy as np
import h5py

path = "orignal dataset\\usps.h5\\usps.h5"

with h5py.File(path, 'r') as hf:
    train = hf.get('train')
    X_tr = train.get('data')[:]
    y_tr = train.get('target')[:]
    test = hf.get('test')
    X_te = test.get('data')[:]
    y_te = test.get('target')[:]
    print(X_tr.shape)
    np.save("dataset\\usps_all.npy", X_tr[0:1000, :].T)
    np.save("dataset\\usps_lab_all.npy", y_tr[0:1000])
    np.hstack
