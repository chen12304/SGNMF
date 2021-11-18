import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from PIL import Image

umist_lab = np.array([], dtype=np.int)
a = loadmat("orignal dataset\\umist_cropped.mat")
print(a)
'''print(a["ImData"].shape)
plt.imshow(a["ImData"][0:644,0].reshape(28,23))
plt.show()'''

for i in range(20):
    print(a["facedat"][0, i].shape)
    for j in range(a["facedat"][0, i].shape[2]):
        im = Image.fromarray(a["facedat"][0, i][:, :, j]).resize((46, 56))
        # im.show()
        temp1 = np.array(im).reshape(46 * 56, -1)
        # print(temp1.shape)
        if j != 0:
            temp = np.hstack((temp, temp1))
        else:
            temp = temp1
    print(temp.shape)
    # temp = a["facedat"][0, i].reshape([92 * 112, -1])
    lab = np.array(i * np.ones(a["facedat"][0, i].shape[2]), dtype=np.int)
    print(lab)
    umist_lab = np.append(umist_lab, lab)
    if i != 0:
        umist = np.hstack((umist, temp))
    else:
        umist = temp
    # plt.imshow(a["facedat"][0, i][:, :, 0])
    # plt.show()
    print(umist_lab.shape, umist.shape)

'''umist = a["ImData"][0:644, :]
umist_lab = np.array(a["ImData"][644, :], dtype=np.int)'''

np.save("dataset\\umist_lab.npy", umist_lab)
np.save("dataset\\umist.npy", umist)
