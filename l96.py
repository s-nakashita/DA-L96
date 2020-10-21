import sys
import logging
import numpy as np
import pandas as pd
from lorenz import step
from obs import add_noise, h_operator
import mlef
import enkf

#logging.config.fileConfig("logging_config.ini")
#logger = logging.getLogger()

global nx, F, dt, dx

nx = 40     # number of points
F  = 8.0    # forcing
dt = 0.05   # time step
#logger.info("nx={} nu={} dt={:7.3e}".format(nx, nu, dt))
print("nx={} F={} dt={:7.3e}".format(nx, F, dt))

x = np.linspace(-2.0, 2.0, nx)
dx = x[1] - x[0]
np.savetxt("x.txt", x)

nmem =   20 # ensemble size
t0off =   8 # initial offset between adjacent members
t0c =    500 # t0 for control
            # t0 for ensemble members
t0m = [t0c + t0off//2 + t0off * i for i in range(-nmem//2, nmem//2)]
t0f = [t0c] + t0m
nt =     1 # number of step per forecast
na =   100 # number of analysis
namax = 1460 # max number of analysis (1 year)
#logger.info("nmem={} t0true={} t0f={}".format(nmem, t0true, t0f))
print("nmem={} t0f={}".format(nmem, t0f))
#logger.info("nt={} na={}".format(nt, na))
print("nt={} na={}".format(nt, na))

sigma = {"linear": 1.0, "quadratic": 8.0e-2, "cubic": 7.0e-4, \
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

def get_true_and_obs(na,nx):
    truth = pd.read_csv("data.csv")
    xt = truth.values.reshape(na,nx)

    obs = pd.read_csv("observation_data.csv")
    y = obs.values.reshape(na,nx)

    return xt, y

def init_ens(nx,nmem,t0c,t0f,dt,F,opt):
    X0c = np.ones(nx)*F
    X0c[nx//2 - 1] += 0.001*F
    tmp = np.zeros_like(X0c)
    maxiter = np.max(np.array(t0f))+1
    if(opt==0): # random
        print("spin up max = {}".format(t0c))
        np.random.seed(514)
        X0 = np.random.normal(0.0,1.0,size=(nx,nmem))
        for j in range(t0c):
            X0c = step(X0c, dt, F)
        X0 += X0c[:,None]
    else: # lagged forecast
        print("spin up max = {}".format(maxiter))
        X0 = np.zeros((nx,nmem))
        tmp = X0c
        for j in range(maxiter):
            tmp = step(tmp, dt, F)
            if j in t0f:
                if t0f.index(j) == 0:
                    X0c = tmp
                else:
                    X0[:,t0f.index(j)-1] = tmp
    return X0c, X0 


def forecast(u, dt, F, kmax, htype):
    for k in range(kmax):
        u = step(u, dt, F)
    if htype["perturbation"] == "etkf" or htype["perturbation"] == "po" \
        or ["perturbation"] == "letkf" or htype["perturbation"] == "srf":
        u[:, 0] = np.mean(u[:, 1:], axis=1)
    return u


def analysis(u, y, rmat, rinv, sig, htype, hist=False, dh=False, model="l96"):
    #logger.info("hist={}".format(hist))
    print("hist={}".format(hist))
    if htype["perturbation"] == "mlef" or htype["perturbation"] == "grad":
        ua, pa, chi2= mlef.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype,
                save_hist=hist, save_dh=dh, model=model)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
    else:
        u_ = np.mean(u[:,1:],axis=1)
        ua, ua_, pa, chi2 = enkf.analysis(u[:, 1:], u_, y, sig, dx, htype, \
            infl = True, loc = True, model=model)
        u[:, 0] = ua_
        u[:, 1:] = ua
    return u, chi2


if __name__ == "__main__":
    op = htype["operator"]
    pt = htype["perturbation"]
    model = "l96"
    rmat, rinv = set_r(nx, sigma[op])
    xt, obs = get_true_and_obs(namax, nx)
    x0c, x0 = init_ens(nx,nmem,t0c,t0f,dt,F,opt=t0off)
    if pt == "mlef" or pt == "grad":
        p0 = x0 - x0c.reshape(-1,1) / np.sqrt(nmem-1)
        x0 = x0c.reshape(-1,1) + p0
    u = np.zeros((nx, nmem+1))
    u[:, 0] = x0c
    u[:, 1:] = x0
    xa = np.zeros((na, nx, nmem+1))
    xf = np.zeros_like(xa)
    e = np.zeros(na)
    chi = np.zeros(na)
    for i in range(na):
        y = obs[i]
        if i == 0:
            #logger.info("first analysis")
            print("first analysis")
            u, chi2 = analysis(u, y, rmat, rinv, sigma[op], htype, \
                hist=True, dh=True, model=model)
        else:
            u, chi2 = analysis(u, y, rmat, rinv, sigma[op], htype, model=model)
        xa[i, :, :] = u
        xf[i, :, :] = u
        chi[i] = chi2
        if i < na-1:
            u = forecast(u, dt, F, nt, htype)
        e[i] = np.sqrt(np.mean((xa[i, :, 0] - xt[i, :])**2))
    np.save("{}_ua_{}_{}.npy".format(model, op, pt), xa)
    np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
    np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
