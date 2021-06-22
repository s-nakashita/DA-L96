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

htype = {"operator":"speed","perturbation":"mlef","gamma":1}
infl = False
infl_parm = 1.15
infl_r = False
tlm = False
l_jhi = False
var = "w+u_noI_rsc"

wmean0 = np.array([2.0,4.0])
wstdv0 = np.array([2.0,2.0])
nmem = 1000
#y = np.array([3.0]) #observation [m/s]
y = np.array([3.0, 1.0]) #observation [m/s]
#y = np.array([1.0, 3.0]) #observation [m/s]
#ytype = ["speed"] #observation type [wind speed]
ytype = ["speed","u"] #observation type [wind speed & u-component]
#ytype = ["u","speed"] #observation type [wind speed & u-component]
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

if htype["perturbation"] == "mlef":
    ## mlef05
    #htype["perturbation"] = htype["perturbation"] + "05"
    if tlm:
        if l_jhi:
            var = var + "_jhi"
        else:
            var = var + "_jhm"
    else:
        var = var + "_fh"
    xf = xf0
    xfc = xc0
    # rescaling 
    pf = (xf - xfc[:, None])/np.sqrt(nmem)
    xf = xfc[:, None] + pf
    # assimilation
    xc, pa, chi2 = analysis(xf, xfc, y, ytype, rmat, rinv, htype, method="CGF", 
                              cgtype=1, maxiter=None, restart=True,
                              infl=infl, infl_parm=infl_parm, 
                              tlm=tlm, l_jhi=l_jhi)
    # rescaling
    pa = pa * np.sqrt(nmem)
    xa = xc[:, None] + pa
    print(np.sqrt(xc[0]**2 + xc[1]**2))
    print(xc[0])
    plot_wind(xf0, xfc, xa, xc, y, sig, ytype, htype, var)
elif htype["perturbation"] == "envar":
    ## envar
    xf = xf0 
    xf_ = np.mean(xf, axis=1)
    dxf = (xf - xf_[:, None])/np.sqrt(nmem-1)
    bmat = dxf @ dxf.transpose()
    print(bmat)
    u, s, vt = np.linalg.svd(dxf)
    print(s.size)
    binv = u @ np.diag(1.0/s**2) @ u.transpose()
    #binv = np.diag(1.0/wstdv0**2)
    print(binv)
    xa = np.zeros_like(xf)
    dy = np.zeros((y.size, nmem))
    for i in range(nmem):
        xa[:, i], chi2 = analysis_var(xf[:, i], binv, y, ytype, rinv, htype, 
            gtol=1e-6, method="CGF", cgtype=1)
        for j in range(y.size):
            op = ytype[j]
            dy[j, i] = obs.dhdx(xf[:, i], op) @ dxf[:, i]
    xa_ = np.mean(xa, axis=1)
    print(np.sqrt(xa_[0]**2 + xa_[1]**2))
    print(xa_[0])
    if infl_r:
        dxa = xa - xa_[:, None]
        sigb = np.sum(np.diag(dy@dy.T))
        sigo = sig*sig*y.size
        print(sigb, sigo)
        if sigb/sigo < 1e-3: #|HPfHt| << |R|
            beta = 0.5
        else:
            beta = 1.0 / (1.0 + np.sqrt(sigo/(sigo + sigb)))
        print("relaxation factor = {}".format(beta))
        dxa = beta * dxa + (1.0 - beta) * dxf * np.sqrt(nmem-1)
        xa = xa_[:, None] + dxa
    plot_wind(xf, xf_, xa, xa_, y, sig, ytype, htype, var)
else:
    ## enkf
    if tlm:
        if l_jhi:
            var = var + "_jhi"
        else:
            var = var + "_jhm"
    else:
        var = var + "_fh"
    xf = xf0 
    xf_ = np.mean(xf, axis=1)
    print(xf_)
    xa, xa_, pa = analysis_enkf(xf, xf_, y, ytype, sig, 0.0, htype,
        infl=infl, infl_parm=infl_parm, tlm=tlm,
        infl_r=infl_r, l_jhi=l_jhi)
    print(np.sqrt(xa_[0]**2 + xa_[1]**2))
    print(xa_[0])
    plot_wind(xf0, xf_, xa, xa_, y, sig, ytype, htype, var)

