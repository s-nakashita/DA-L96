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

def f_ls(x0, g, alpha, calc_j, *args):
    f = np.zeros(len(alpha))
    for a in alpha:
        x = x0 + a*g
        f[alpha.index(a)] = calc_j(x, *args)
    return f

htype = {"operator":"speed","perturbation":"mlef","gamma":1}
pert = ["mlef", "grad", "mlef05", "mlefw", "mlefb", "mleft"]
alpha = np.linspace(-0.15, 0.15, 101, endpoint=True).tolist()
wmean = np.array([2.0,4.0])
wstdv = np.array([2.0,2.0])
nmem = 1000
y = np.array([3.0]) #observation [m/s]
sig = 0.3 #observation error [m/s]
rmat = np.array([1.0 / sig]).reshape(-1,1)
rinv = np.array([1.0 / sig / sig]).reshape(-1,1)   

np.random.seed(514)
wind = random.multivariate_normal(wmean, np.diag(wstdv), size=nmem)
xf = wind.transpose()
xfc = wmean
xf_ = np.mean(xf, axis=1)
pf = xf - xfc[:, None]
dxf = (xf - xf_[:, None])/np.sqrt(nmem-1)

x0 = np.zeros(nmem)
# mlef
dh = obs.h_operator(xf, htype["operator"], htype["gamma"]) - obs.h_operator(xfc, htype["operator"], htype["gamma"])[:, None]
zmat = rmat @ dh
tmat, heinv, condh = precondition(zmat)
gmat = pf@tmat
args = (xfc, pf, y, tmat, gmat, heinv, rinv, htype)
g = calc_grad_j(x0, *args)
alpha = np.linspace(-0.05, 0.05, 101, endpoint=True).tolist()
f = f_ls(x0, -g, alpha, calc_j, *args)
plt.plot(alpha, f)
plt.xticks(alpha[::10])
plt.xlabel(r"step_length $\alpha$")
plt.title(r"$f(\zeta_0 - \alpha\nabla f)$")
plt.savefig("wind_assim/mlef_fls_z.png")
plt.close()
# grad
htype["perturbation"] = "grad"
dh = obs.dhdx(xfc, htype["operator"], htype["gamma"]) @ pf
zmat = rmat @ dh 
tmat, heinv, condh = precondition(zmat)
gmat = pf@tmat
args = (xfc, pf, y, tmat, gmat, heinv, rinv, htype)
g = calc_grad_j(x0, *args)
alpha = np.linspace(-0.05, 0.05, 101, endpoint=True).tolist()
f = f_ls(x0, -g, alpha, calc_j, *args)
plt.plot(alpha, f)
plt.xticks(alpha[::10])
plt.xlabel(r"step_length $\alpha$")
plt.title(r"$f(\zeta_0 - \alpha\nabla f)$")
plt.savefig("wind_assim/grad_fls_z.png")
plt.close()
# mlef05
htype["perturbation"] = "mlef"
dh = obs.h_operator(xf, htype["operator"], htype["gamma"]) - obs.h_operator(xfc, htype["operator"], htype["gamma"])[:, None]
zmat = rmat @ dh
tmat, heinv, condh = precondition(zmat)
gmat = pf@tmat
args = (xfc, pf, y, tmat, gmat, heinv, rinv, htype)
g = calc_grad_j05(x0, *args)
alpha = np.linspace(-0.9, 1.2, 101, endpoint=True).tolist()
f = f_ls(x0, -g, alpha, calc_j05, *args)
plt.plot(alpha, f)
plt.xticks(alpha[::10])
plt.xlabel(r"step_length $\alpha$")
plt.title(r"$f(\zeta_0 - \alpha\nabla f)$")
plt.savefig("wind_assim/mlef05_fls_z.png")
plt.close()
# mlefw
args = (xfc, pf, y, rinv, htype)
g = calc_grad_j2(x0, *args)
alpha = np.linspace(-3.0e-4, 3.0e-4, 101, endpoint=True).tolist()
f = f_ls(x0, -g, alpha, calc_j2, *args)
plt.plot(alpha, f)
plt.xticks(alpha[::20])
plt.xlabel(r"step_length $\alpha$")
plt.title(r"$f(\mathbf{w}_0 - \alpha\nabla f)$")
plt.savefig("wind_assim/mlefw_fls_z.png")
plt.close()
# mlefb
htype["perturbation"] = "mlefb"
eps = 1e-4
emat = xf_[:, None] + eps*dxf
hemat = obs.h_operator(emat, htype["operator"], htype["gamma"])
dh = (hemat - np.mean(hemat, axis=1)[:, None]) / eps 
zmat = rmat @ dh 
tmat, tinv, heinv, condh = precb(zmat)
args = (xf_, dxf, y, precb, eps, rmat, htype)
g = calc_grad_jb(x0, *args)
alpha = np.linspace(-0.2, 0.2, 101, endpoint=True).tolist()
f = f_ls(x0, -g, alpha, calc_jb, *args)
plt.plot(alpha, f)
plt.xticks(alpha[::10])
plt.xlabel(r"step_length $\alpha$")
plt.title(r"$f(\mathbf{w}_0 - \alpha\nabla f)$")
plt.savefig("wind_assim/mlefb_fls_z.png")
plt.close()
# mleft
htype["perturbation"] = "mleft"
tmat = np.eye(nmem)
tinv = tmat
np.save("tmat.npy",tmat)
np.save("tinv.npy",tinv)
emat = xf_[:, None] + np.sqrt(nmem-1)*dxf@tmat
hemat = obs.h_operator(emat, htype["operator"], htype["gamma"])
dh = (hemat - np.mean(hemat, axis=1)[:, None]) @ tinv / np.sqrt(nmem-1)
tmat, tinv = calc_hess(dh, rinv)
args = (xf_, dxf, y, calc_hess, rinv, htype)
g = calc_grad_jt(x0, *args)
alpha = np.linspace(-0.2, 0.2, 101, endpoint=True).tolist()
f = f_ls(x0, -g, alpha, calc_jt, *args)
plt.plot(alpha, f)
plt.xticks(alpha[::10])
plt.xlabel(r"step_length $\alpha$")
plt.title(r"$f(\mathbf{w}_0 - \alpha\nabla f)$")
plt.savefig("wind_assim/mleft_fls_z.png")
plt.close()