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
pert = ["mlef", "grad", "mlef05", "mlefw", "mlefb", "mleft"]
sig_b = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
a = np.linspace(0.67, 2.0, 10)
sig_o = [5.0, 3.0, 1.0, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01]
cg = np.zeros((len(pert),len(sig_o)))
wmean0 = np.array([2.0,4.0])
wstdv0 = np.array([2.0,2.0])
nmem = 1000
y = np.array([3.0]) #observation [m/s]
sig = 0.3 #observation error [m/s]
rmat = np.array([1.0 / sig]).reshape(-1,1)
rinv = np.array([1.0 / sig / sig]).reshape(-1,1)   

#for i in range(len(sig_b)):
#for i in range(len(a)):
for i in range(len(sig_o)):
    var = "r_{}".format(i)
    sig = sig_o[i] #observation error [m/s]
    rmat = np.array([1.0 / sig]).reshape(-1,1)
    rinv = np.array([1.0 / sig / sig]).reshape(-1,1)
    np.random.seed(514)
    #wmean = wmean0 * a[i]
    wmean = wmean0
    #wstdv = np.array([sig_b[i],sig_b[i]])
    wstdv = wstdv0
    wind = random.multivariate_normal(wmean0, np.diag(wstdv), size=nmem)
    xf = wind.transpose()
    xfc = wmean
    xf_ = np.mean(xf, axis=1)
    pf = xf - xfc[:, None]
    dxf = (xf - xf_[:, None])/np.sqrt(nmem-1)
    x0 = np.zeros(nmem)
    # mlef
    xc, pa, chi2, ds, condh = analysis(xf, xfc, y, rmat, rinv, htype)
    xa = xc[:, None] + pa
    plot_wind(xf, xfc, xa, xc, y, sig, htype, "mlef", var)
    dh = obs.h_operator(xf, htype["operator"], htype["gamma"]) - obs.h_operator(xfc, htype["operator"], htype["gamma"])[:, None]
    zmat = rmat @ dh
    tmat, heinv, condh = precondition(zmat)
    gmat = pf@tmat
    args = (xfc, pf, y, tmat, gmat, heinv, rinv, htype)
    cg[0, i] = spo.check_grad(calc_j, calc_grad_j, x0, *args)
    # grad
    htype["perturbation"] = "grad"
    xc, pa, chi2, ds, condh = analysis(xf, xfc, y, rmat, rinv, htype)
    xa = xc[:, None] + pa
    plot_wind(xf, xfc, xa, xc, y, sig, htype, "grad", var)
    dh = obs.dhdx(xfc, htype["operator"], htype["gamma"]) @ pf
    zmat = rmat @ dh 
    tmat, heinv, condh = precondition(zmat)
    gmat = pf@tmat
    args = (xfc, pf, y, tmat, gmat, heinv, rinv, htype)
    cg[1, i] = spo.check_grad(calc_j, calc_grad_j, x0, *args)
    # mlef05
    htype["perturbation"] = "mlef"
    xc, pa, chi2 = analysis05(xf, xfc, y, rmat, rinv, htype)
    xa = xc[:, None] + pa
    plot_wind(xf, xfc, xa, xc, y, sig, htype, "mlef05", var)
    dh = obs.h_operator(xf, htype["operator"], htype["gamma"]) - obs.h_operator(xfc, htype["operator"], htype["gamma"])[:, None]
    zmat = rmat @ dh
    tmat, heinv, condh = precondition(zmat)
    gmat = pf@tmat
    args = (xfc, pf, y, tmat, gmat, heinv, rinv, htype)
    cg[2, i] = spo.check_grad(calc_j05, calc_grad_j05, x0, *args)
    # mlefw
    xc, pa, chi2 = analysis2(xf, xfc, y, rmat, rinv, htype)
    xa = xc[:, None] + pa
    plot_wind(xf, xfc, xa, xc, y, sig, htype, "mlefw", var)
    args = (xfc, pf, y, rinv, htype)
    cg[3, i] = spo.check_grad(calc_j2, calc_grad_j2, x0, *args)
    # mlefb
    htype["perturbation"] = "mlefb"
    xa, xa_, pa, chi2, ds, condh = analysisb(xf, xf_, y, rmat, rinv, htype)
    plot_wind(xf, xf_, xa, xa_, y, sig, htype, "mlefb", var)
    eps = 1e-4
    emat = xf_[:, None] + eps*dxf
    hemat = obs.h_operator(emat, htype["operator"], htype["gamma"])
    dh = (hemat - np.mean(hemat, axis=1)[:, None]) / eps 
    zmat = rmat @ dh 
    tmat, tinv, heinv, condh = precb(zmat)
    args = (xf_, dxf, y, precb, eps, rmat, htype)
    cg[4, i] = spo.check_grad(calc_jb, calc_grad_jb, x0, *args)
    # mleft
    htype["perturbation"] = "mleft"
    xa, xa_, pa, chi2, ds, condh = analysist(xf, xf_, y, rmat, rinv, htype)
    plot_wind(xf, xf_, xa, xa_, y, sig, htype, "mleft", var)
    tmat = np.eye(nmem)
    tinv = tmat
    np.save("tmat.npy",tmat)
    np.save("tinv.npy",tinv)
    emat = xf_[:, None] + np.sqrt(nmem-1)*dxf@tmat
    hemat = obs.h_operator(emat, htype["operator"], htype["gamma"])
    dh = (hemat - np.mean(hemat, axis=1)[:, None]) @ tinv / np.sqrt(nmem-1)
    tmat, tinv = calc_hess(dh, rinv)
    args = (xf_, dxf, y, calc_hess, rinv, htype)
    cg[5, i] = spo.check_grad(calc_jt, calc_grad_jt, x0, *args)
fig, ax = plt.subplots()
#width = 0.05*np.array(sig_b)
#width = 0.1
width = 0.05*np.array(sig_o)
#xaxis = np.array(sig_b) - 2.5*width
#xaxis = np.arange(a.size) - 2.5*width
xaxis = np.array(sig_o) - 2.5*width
for j in range(len(pert)):
    ax.bar(xaxis, cg[j], width=width, label=pert[j])
    xaxis += width
#ax.set_xticks(sig_b)
#ax.set_xticks(np.arange(a.size))
ax.set_xticks(sig_o)
ax.set_xscale("log")
ax.set_yscale("log")
#ax.set_xlabel("background error stdv.")
#ax.set_xlabel("ctrl. location")
ax.set_xlabel("observation error stdv.")
ax.set_ylabel("error 2-norm")
ax.set_title("estimated gradient - finite difference approximation")
ax.legend(ncol=2)
fig.savefig("check_grad_r.png")