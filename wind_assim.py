import numpy as np
from numpy import random
import scipy.optimize as spo
import matplotlib.pyplot as plt
import obs
from plot_wind import plot_wind
from mlef import analysis
from mlef import precondition, calc_j, calc_grad_j
from mlef05 import analysis as analysis05
from mlef05 import calc_j as calc_j05
from mlef05 import calc_grad_j as calc_grad_j05
from mlef2 import analysis as analysis2
from mlef2 import calc_j as calc_j2
from mlef2 import calc_grad_j as calc_grad_j2
from mlefb import analysis as analysisb
from mlefb import precondition as precb
from mlefb import calc_j as calc_jb
from mlefb import calc_grad_j as calc_grad_jb
from mleft import analysis as analysist
from mleft import calc_hess
from mleft import calc_j as calc_jt
from mleft import calc_grad_j as calc_grad_jt

htype = {"operator":"speed","perturbation":"mlef","gamma":1}
wmean0 = np.array([2.0,4.0])
wstdv0 = np.array([2.0,2.0])
nmem = 1000
y = np.array([3.0]) #observation [m/s]
sig = 0.3 #observation error [m/s]
rmat = np.array([1.0 / sig]).reshape(-1,1)
rinv = np.array([1.0 / sig / sig]).reshape(-1,1)   

np.random.seed(514)
wind = random.multivariate_normal(wmean0, np.diag(wstdv0), size=nmem)
xf = wind.transpose()
xfc = wmean0
xf_ = np.mean(xf, axis=1)
pf = xf - xfc[:, None]
dxf = (xf - xf_[:, None])/np.sqrt(nmem-1)
x0 = np.zeros(nmem)
# mlef05
var = "tnc"
htype["perturbation"] = "mlefw"
xc, pa, chi2 = analysis2(xf, xfc, y, rmat, rinv, htype, maxiter=200)#, method="CGF", 
#                          cgtype=3, maxiter=5, restart=True)
xa = xc[:, None] + pa
print(np.sqrt(xc[0]**2 + xc[1]**2))
plot_wind(xf, xfc, xa, xc, y, sig, htype, "mlefw", var)
