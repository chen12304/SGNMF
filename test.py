from function import *
from advanced_agolorithm import cfsfdp

a = cfsfdp(data_umist.T, r=3700/255, classnum=10)
print(acc(lab_umist, a), nmi(lab_umist, a))
