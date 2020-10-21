import sys
import logging
import numpy as np
from burgers import step
from obs import add_noise, h_operator
import mlef
import enkf

#logging.config.fileConfig("logging_config.ini")
#logger = logging.getLogger()

global nx, nu, dt, dx

nx = 81     # number of points
nu = 0.05   # diffusion
dt = 0.0125 # time step
#logger.info("nx={} nu={} dt={:7.3e}".format(nx, nu, dt))
print("nx={} nu={} dt={:7.3e}".format(nx, nu, dt))

x = np.linspace(-2.0, 2.0, nx)
dx = x[1] - x[0]
np.savetxt("x.txt", x)

nmem =    4 # ensemble size
t0off =   8 # initial offset between adjacent members
t0true = 20 # t0 for true
t0c =    60 # t0 for control
            # t0 for ensemble members
t0m = [t0c + t0off//2 + t0off * i for i in range(-nmem//2, nmem//2)]
t0f = [t0c] + t0m
nt =     20 # number of step per forecast
na =     20 # number of analysis
#logger.info("nmem={} t0true={} t0f={}".format(nmem, t0true, t0f))
print("nmem={} t0true={} t0f={}".format(nmem, t0true, t0f))
#logger.info("nt={} na={}".format(nt, na))
print("nt={} na={}".format(nt, na))

sigma = {"linear": 8.0e-2, "quadratic": 8.0e-2, "cubic": 7.0e-4, \
    "quadratic-nodiff": 8.0e-2, "cubic-nodiff": 7.0e-4,}
htype = {"operator": "linear", "perturbation": "mlef"}
if len(sys.argv) > 1:
    htype["operator"] = sys.argv[1]
if len(sys.argv) > 2:
    htype["perturbation"] = sys.argv[2]
#logger.info("htype={} sigma={}".format(htype, sigma[htype["operator"]]))
print("htype={} sigma={}".format(htype, sigma[htype["operator"]]))


def set_r(nx, sigma):
    rmat = np.diag(np.ones(nx) / sigma)
    rinv = rmat.transpose() @ rmat
    return rmat, rinv


def gen_true(x, dt, nu, t0true, t0f, nt, na):
    nx = x.size
    nmem = len(t0f)
    u = np.zeros_like(x)
    u[0] = 1
    dx = x[1] - x[0]
    ut = np.zeros((na, nx))
    u0 = np.zeros((nx, nmem))
    for k in range(t0true):
        u = step(u, dx, dt, nu)
    ut[0, :] = u
    for i in range(na-1):
        for k in range(nt):
            u = step(u, dx, dt, nu)
            j = (i + 1) * nt + k
            if j in t0f:
                u0[:, t0f.index(j)] = u
        ut[i+1, :] = u
    return ut, u0


def gen_obs(u, sigma, op):
    y = h_operator(add_noise(u, sigma), op)
    return y


def forecast(u, dx, dt, nu, kmax, htype):
    for k in range(kmax):
        u = step(u, dx, dt, nu)
    if htype["perturbation"] == "etkf" or htype["perturbation"] == "po" \
        or ["perturbation"] == "letkf" or htype["perturbation"] == "srf":
        u[:, 0] = np.mean(u[:, 1:], axis=1)
    return u


def analysis(u, y, rmat, rinv, sig, htype, hist=False, dh=False, model="z08"):
    #logger.info("hist={}".format(hist))
    print("hist={}".format(hist))
    if htype["perturbation"] == "mlef" or htype["perturbation"] == "grad":
        ua, pa, chi2= mlef.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype,
                save_hist=hist, save_dh=dh, model=model)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
    else:
        u_ = np.mean(u[:,1:],axis=1)
        ua, ua_, pa, chi2 = enkf.analysis(u[:, 1:], u_, y, sig, dx, htype, model=model)
        u[:, 0] = ua_
        u[:, 1:] = ua
    return u, chi2


if __name__ == "__main__":
    op = htype["operator"]
    pt = htype["perturbation"]
    model = "z08"
    rmat, rinv = set_r(nx, sigma[op])
    ut, u = gen_true(x, dt, nu, t0true, t0f, nt, na)
    if pt == "mlef" or pt == "grad":
        p0 = u[:, 1:] - u[:, 0].reshape(-1,1) / np.sqrt(nmem-1)
        u[:, 1:] = u[:, 0].reshape(-1,1) + p0
    ua = np.zeros((na, nx, nmem+1))
    uf = np.zeros_like(ua)
    e = np.zeros(na)
    chi = np.zeros(na)
    for i in range(na):
        y = gen_obs(ut[i,], sigma[op], op)
        if i == 3:
            #logger.info("first analysis")
            print("first analysis")
            u, chi2 = analysis(u, y, rmat, rinv, sigma[op], htype, \
                hist=True, dh=True, model=model)
        else:
            u, chi2 = analysis(u, y, rmat, rinv, sigma[op], htype, model=model)
        ua[i, :, :] = u
        uf[i, :, :] = u
        chi[i] = chi2
        if i < na-1:
            u = forecast(u, dx, dt, nu, nt, htype)
        e[i] = np.sqrt(np.mean((ua[i, :, 0] - ut[i, :])**2))
    np.save("ut.npy", ut)
    np.save("{}_uf_{}_{}.npy".format(model, op, pt), uf)
    np.save("{}_ua_{}_{}.npy".format(model, op, pt), ua)
    np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
    np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
