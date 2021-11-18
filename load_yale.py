import os
from PIL import Image
import numpy as np


def get_yale(file_path):
    im_array = np.array([])
    lab = np.array([], dtype=np.int)
    for file in os.listdir(file_path):
        im = Image.open(file_path + '\\' + file)
        im = np.array(im)
        # print(im)
        # print(im.shape)
        im_array = np.append(im_array, im / 255)  # 归一化
        print(file)
        # print(lab)
    return im_array.reshape(-1, 243 * 320).T, lab

path = "orignal dataset\\yalefaces\\yalefaces"
a,b = get_yale(path)
print(a.shape)
def load_lab(lab, num):
    if num > 0:
        return load_lab(np.append(lab, num * np.ones(11, dtype=int)), num - 1)
    else:
        return lab
yale_lab = np.array([],dtype = np.int)
yale_lab = load_lab(yale_lab,15)
print(yale_lab)
np.save("dataset\\yale.npy", a)
np.save("dataset\\yale_lab.npy", yale_lab)
