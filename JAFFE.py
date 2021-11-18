import os
from PIL import Image
import numpy as np


def get_JAFFE(file_path):
    im_array = np.array([])
    lab = np.array([], dtype=np.int)
    for file in os.listdir(file_path):
        im = Image.open(file_path + '\\' + file)
        im = np.array(im)
        # print(im)
        print(im.shape)
        im_array = np.append(im_array, im / 255)  # 归一化
        print(file)
        lab = np.append(lab, (file[0:file.find('.')]))
        print(lab)
    return im_array.reshape(-1, 256 * 256).T, lab
path = "orignal dataset\\JAFFE\\JAFFE"
Jaffe,jaffe_lab=get_JAFFE(path)
jaffe_lab_n=np.zeros(213)
jaffe_lab_n += (jaffe_lab=="KL")
jaffe_lab_n+=(2*(jaffe_lab=="KM"))
jaffe_lab_n+=(3*(jaffe_lab=="KR"))
jaffe_lab_n+=(4*(jaffe_lab=="MK"))
jaffe_lab_n+=(5*(jaffe_lab=="NA"))
jaffe_lab_n+=(6*(jaffe_lab=="NM"))
jaffe_lab_n+=(7*(jaffe_lab=="TM"))
jaffe_lab_n+=(8*(jaffe_lab=="UY"))
jaffe_lab_n+=(9*(jaffe_lab=="YM"))
jaffe_lab_n = np.array(jaffe_lab_n,dtype = np.int)
print(jaffe_lab_n)
np.save("dataset\\JAFFE.npy", Jaffe)
np.save("dataset\\JAFFE_lab.npy", jaffe_lab_n)