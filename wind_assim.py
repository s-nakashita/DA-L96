import numpy as np
from numpy import random
import scipy.optimize as spo
import matplotlib.pyplot as plt
import obs
from plot_wind import plot_wind
from mlef_wind import analysis
from mlef_wind import precondition, calc_j, calc_grad_j
from enkf_wind import analysis as analysis_enkf
from var_wind import analysis as analysis_var
from hyvar import analysis as analysis_hyvar

htype = {"operator":"speed","perturbation":"grad","gamma":1}
tlm = True
var = "wind+u"

wmean0 = np.array([2.0,4.0])
wstdv0 = np.array([2.0,2.0])
nmem = 1000
#y = np.array([3.0]) #observation [m/s]
y = np.array([3.0, 1.0]) #observation [m/s]
ytype = ["speed","u"] #observation type [wind speed & u-component]
sig = 0.3 #observation error [m/s]
#rmat = np.array([1.0 / sig]).reshape(-1,1)
#rinv = np.array([1.0 / sig / sig]).reshape(-1,1)   
rmat = np.eye(y.size) / sig
rinv = np.eye(y.size) / sig / sig

np.random.seed(514)
wind = random.multivariate_normal(wmean0, np.diag(wstdv0), size=nmem)
xf0 = wind.transpose()
xc0 = wmean0
x0 = np.zeros(nmem)

if htype["perturbation"] == "mlef" or htype["perturbation"] == "grad":
    ## mlef05
    htype["perturbation"] = htype["perturbation"] + "05"
    xf = xf0
    xfc = xc0
    # assimilation
    xc, pa, chi2 = analysis(xf, xfc, y, ytype, rmat, rinv, htype, method="LBFGS", 
                              cgtype=None, maxiter=None, restart=True)
    xa = xc[:, None] + pa
    print(np.sqrt(xc[0]**2 + xc[1]**2))
    plot_wind(xf0, xfc, xa, xc, y, sig, htype, var)
elif htype["perturbation"] == "envar":
    ## envar
    xf = xf0 
    xf_ = np.mean(xf, axis=1)
    dxf = (xf - xf_[:, None])/np.sqrt(nmem-1)
    u, s, vt = np.linalg.svd(dxf)
    binv = u[:, :nmem] @ np.diag(1.0/s**2) @ u[:, :nmem].transpose()
    binv = np.diag(1.0/wstdv0**2)
    print(binv)
    xa = np.zeros_like(xf)
    for i in range(nmem):
        xa[:, i], chi2 = analysis_var(xf[:, i], binv, y, ytype, rinv, htype, 
            gtol=1e-6, method="LBFGS", cgtype=None)
    xa_ = np.mean(xa, axis=1)
    print(np.sqrt(xa_[0]**2 + xa_[1]**2))
    plot_wind(xf, xf_, xa, xa_, y, sig, htype, var)
else:
    ## enkf
    if tlm:
        var = var + "_jhi"
    else:
        var = var + "_fh"
    xf = xf0 
    xf_ = np.mean(xf, axis=1)
    xa, xa_, pa = analysis_enkf(xf, xf_, y, ytype, sig, 0.0, htype, tlm=tlm)
    print(np.sqrt(xa_[0]**2 + xa_[1]**2))
    plot_wind(xf0, xf_, xa, xa_, y, sig, htype, var)

