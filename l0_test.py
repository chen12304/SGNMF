import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import datasets

# print(datasets.list_datasets())
a = np.arange(0, 20).reshape([2, 2, 5])
print(a)
a = a.reshape(4, 5)
print(a)

'''x = np.linspace(0,1,1001)
y1 = np.sin(np.arctan(x**2/0.1**2))
y2 = (np.exp(2*x*x/0.1/0.1)-1)/(np.exp(2*x*x/0.1/0.1)+1)
plt.plot(x,y1)
plt.plot(x,y2)'''
# plt.show()

