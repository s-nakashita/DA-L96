import sys
import os
import logging
import numpy as np
import pandas as pd
from lorenz import step
from obs import add_noise, h_operator
#from obs2 import Obs
import mlef
import mlefb
import mleft
import mlef05
import mlef2 as mlefw
import mlef3
import enkf

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger()

global nx, F, dt, dx

nx = 40     # number of points
F  = 8.0    # forcing
dt = 0.05 / 6  # time step (=1 hour)
logger.info("nx={} F={} dt={:7.3e}".format(nx, F, dt))
#print("nx={} F={} dt={:7.3e}".format(nx, F, dt))

x = np.linspace(-2.0, 2.0, nx)
dx = x[1] - x[0]
np.savetxt("x.txt", x)

nmem =   20 # ensemble size
t0off =   8 # initial offset between adjacent members
t0c =    500 # t0 for control
            # t0 for ensemble members
t0m = [t0c + t0off//2 + t0off * i for i in range(-nmem//2, nmem//2)]
t0f = [t0c] + t0m
nt =     6 # number of step per forecast (=6 hour)
na =   100 # number of analysis
namax = 1460 # max number of analysis (1 year)

sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "linear-nodiff": 1.0, "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0, "abs":1.0}
#sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
#    "quadratic-nodiff": 1.0, "cubic-nodiff": 1.0, "test":1.0}
htype = {"operator": "linear", "perturbation": "mlef", "gamma": 1}
linf = False
lloc = False
ltlm = True
if len(sys.argv) > 1:
    htype["operator"] = sys.argv[1]
if len(sys.argv) > 2:
    htype["perturbation"] = sys.argv[2]
if len(sys.argv) > 3:
    na = int(sys.argv[3])
if len(sys.argv) > 4:
    infl_parm = float(sys.argv[4])
    if infl_parm > 0.0:
        linf = True
    #if sys.argv[4] == "T":
    #    linf = True
if len(sys.argv) > 5:
    if sys.argv[5] == "T":
        lloc = True
if len(sys.argv) > 6:
    if sys.argv[6] == "F":
        ltlm = False
maxiter = None
if len(sys.argv) > 7:
    #htype["gamma"] = int(sys.argv[7])
    #maxiter = int(sys.argv[6])
    nt = int(sys.argv[7])
method = "CGF"
method_short = "cgf_fr"
cgtype = 1
if len(sys.argv) > 8:
    method_short = sys.argv[8]
    if method_short == "bg":
        method = "BFGS"
    elif method_short == "cg":
        method = "CG"
    elif method_short == "nm":
        method = "Nelder-Mead"
    elif method_short == "gd":
        method = "GD"
    elif method_short == "pw":
        method = "Powell"
    elif method_short == "gd":
        method = "GD"
    elif method_short == "ncg":
        method = "Newton-CG"
    elif method_short == "dog":
        method = "dogleg"
    elif method_short[0:3] == "cgf":
        method = "CGF"
        if method_short[4:] == "fr":
            cgtype = 1
        elif method_short[4:] == "pr":
            cgtype = 2
        elif method_short[4:] == "prb":
            cgtype = 3
restart = False
maxrest = 20 # outer-loop iteration
if len(sys.argv) > 9:
    if sys.argv[9] == "T":
        restart = True
    elif sys.argv[9] == "F":
        restart = False
logger.info("nmem={} t0f={}".format(nmem, t0f))
#print("nmem={} t0f={}".format(nmem, t0f))
logger.info("nt={} na={}".format(nt, na))
#print("nt={} na={}".format(nt, na))
logger.info("htype={} sigma={}".format(htype, sigma[htype["operator"]]))
#print("htype={} sigma={}".format(htype, sigma[htype["operator"]]))
logger.info("inflation={} localization={} TLM={}".format(linf,lloc,ltlm))
#print("inflation={} localization={} TLM={}".format(linf,lloc,ltlm))
logger.info("inflation parameter={}".format(infl_parm))
#print("inflation parameter={}".format(infl_parm))
logger.info("minimization method={} restart={}".format(method, restart))

#obs = Obs(htype["operator"], sigma[htype["operator"]])

def set_r(nx, sigma):
    rmat = np.diag(np.ones(nx) / sigma)
    rinv = rmat.transpose() @ rmat
    return rmat, rinv

def get_true_and_obs(na, nx, sigma, op, gamma=1):
    truth = pd.read_csv("data.csv")
    xt = truth.values.reshape(na,nx)

    #obs = pd.read_csv("observation_data.csv")
    #y = obs.values.reshape(na,nx)
    f = "obs_{}.npy".format(op)
    if not os.path.isfile(f):
        seed = 514
        y = add_noise(h_operator(xt, op, gamma), sigma) #, seed=seed)
        np.save(f, y)
    else:
        y = np.load(f)
    #y = np.load("obs_{}.npy".format(op))
    #y = np.zeros((na, nx, 2))
    #obsloc = np.arange(nx)
    #for k in range(na):
    #    y[k,:,0] = obsloc[:]
    #    y[k,:,1] = obs.add_noise(obs.h_operator(obsloc, xt[k]))

    return xt, y

def init_ens(nx,nmem,t0c,t0f,dt,F,pt,opt):
    X0 = np.zeros((nx, nmem))
    X0c = np.ones(nx)*F
    X0c[nx//2 - 1] += 0.001*F
    tmp = np.zeros_like(X0c)
    maxiter = np.max(np.array(t0f))+1
    if(opt==0): # random
        logger.info("spin up max = {}".format(t0c))
        np.random.seed(514)
        for j in range(t0c):
            #X0 = step(X0, dt, F)
            X0c = step(X0c, dt, F)
        logger.debug("X0c={}".format(X0c))
        #if pt == "mlef" or pt == "grad" or pt == "mlef05" or pt == "grad05" or pt == "mlefw" or pt == "mlef3":
        #    X0[:, 0] = X0c
        #    X0[:, 1:] = np.random.normal(0.0,1.0,size=(nx,nmem-1)) + X0c[:, None]
        #else:
        X0 = np.random.normal(0.0,1.0,size=(nx,nmem)) + X0c[:, None]
        #for j in range(t0c):
        #    X0 = step(X0, dt, F)
        #    X0c = step(X0c, dt, F)
    else: # lagged forecast
        logger.info("spin up max = {}".format(maxiter))
        X0 = np.zeros((nx,nmem))
        tmp = X0c
        for j in range(maxiter):
            tmp = step(tmp, dt, F)
            if j in t0f:
                if t0f.index(j) == 0:
                    X0[:,t0f.index(j)] = tmp
    return X0


def forecast(u, dt, F, kmax, htype):
    for k in range(kmax):
        u = step(u, dt, F)
    #if htype["perturbation"] == "etkf" or htype["perturbation"] == "po" \
    #    or ["perturbation"] == "letkf" or htype["perturbation"] == "srf":
    #    u[:, 0] = np.mean(u[:, 1:], axis=1)
    return u


def analysis(u, y, rmat, rinv, sig, htype, hist=False, dh=False, \
    method="CG", cgtype=None,\
    maxiter=None, restart=True, maxrest=20, 
    infl=False, loc=False, tlm=True, infl_parm=1.0,\
    model="l96", icycle=0):
    logger.info("hist={}".format(hist))
    #print("hist={}".format(hist))
    if htype["perturbation"] == "mlef" or htype["perturbation"] == "grad":
    #    ua, pa, chi2, ds, condh = mlef.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype, \
    #        method=method, maxiter=maxiter, save_hist=hist, save_dh=dh, \
    #        infl=infl, loc=loc, infl_parm=infl_parm, model=model, icycle=icycle)
    #    u[:, 0] = ua
    #    u[:, 1:] = ua[:, None] + pa
    #elif htype["perturbation"] == "mlef05" or htype["perturbation"] == "grad05":
        ua, pa, chi2 = mlef05.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype,
            method=method, cgtype=cgtype,
            maxiter=maxiter, restart=restart, maxrest=maxrest, 
            save_hist=hist, save_dh=dh, \
            infl=infl, loc=loc, infl_parm=infl_parm, model=model, icycle=icycle)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
        ds = 0.0
        condh = 0.0
    elif htype["perturbation"] == "mlefw":
        ua, pa, chi2 = mlefw.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype,
            method=method, maxiter=maxiter, save_hist=hist, save_dh=dh, \
            infl=infl, loc=loc, infl_parm=infl_parm, model=model, icycle=icycle)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
        ds = 0.0
        condh = 0.0
    elif htype["perturbation"] == "mlef3":
        ua, pa, chi2, ds = mlef3.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype,
            method=method, maxiter=maxiter, save_hist=hist, save_dh=dh, \
            infl=infl, loc=loc, infl_parm=infl_parm, model=model, icycle=icycle)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
        condh = 0.0
    elif htype["perturbation"] == "mlefb":
        u_ = np.mean(u, axis=1)
        ua, ua_, pa, chi2, ds, condh = mlefb.analysis(u, u_, y, rmat, rinv, htype, \
            method=method, maxiter=maxiter, save_hist=hist, save_dh=dh, \
            infl = infl, loc = loc, infl_parm = infl_parm, model=model, icycle=icycle)
        u[:, :] = ua
        #u[:, 1:] = ua[:, None] + pa
    elif htype["perturbation"] == "mleft":
        u_ = np.mean(u, axis=1)
        ua, ua_, pa, chi2, ds, condh = mleft.analysis(u, u_, y, rmat, rinv, htype, \
            method=method, maxiter=maxiter, save_hist=hist, save_dh=dh, \
            infl = infl, loc = loc, infl_parm = infl_parm, model=model, icycle=icycle)
        u[:, :] = ua
        #u[:, 1:] = ua[:, None] + pa
    else:
        u_ = np.mean(u,axis=1)
        ua, ua_, pa, chi2, ds, condh = enkf.analysis(u, u_, y, sig, dx, htype, \
            infl = infl, loc = loc, tlm=tlm, infl_parm = infl_parm, \
            save_dh=dh, model=model, icycle=icycle)
        u[:, :] = ua
        #u[:, 0] = ua_
        #u[:, 1:] = ua
    return u, pa, chi2, ds

def plot_initial(uc, u, ut, pt, model):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = np.arange(ut.size) + 1
    ax.plot(x, ut, label="true")
    if pt == "mlef" or pt == "grad":
        ax.plot(x, uc, label="control")
    else:
        ax.plot(x, np.mean(u, axis=1), label="mean")
    for i in range(u.shape[1]):
        ax.plot(x, u[:,i], linestyle="--", label="mem{}".format(i+1))
    ax.set(xlabel="points", ylabel="X", title="initial state")
    ax.set_xticks(x[::5])
    ax.set_xticks(x, minor=True)
    ax.legend(ncol=2)
    fig.savefig("{}_initial_{}.png".format(model, pt))

if __name__ == "__main__":
    op = htype["operator"]
    pt = htype["perturbation"]
    ga = htype["gamma"]
    model = "l96"
    rmat, rinv = set_r(nx, sigma[op])

    xt, obs = get_true_and_obs(namax, nx, sigma[op], op, ga)
    logger.info("obs.shape={}".format(obs.shape))
    #np.save("{}_ut.npy".format(model), xt[:na,:])
    x0 = init_ens(nx,nmem,t0c,t0f,dt,F,pt,opt=0)
    if pt == "mlef" or pt == "grad" or pt == "mlef05" or pt == "grad05" or pt == "mlefw" or pt == "mlef3":
        x0c = x0[:, 0]
        pf = (x0[:, 1:] - x0c[:,None]) @ (x0[:, 1:] - x0c[:,None]).T
        #plot_initial(x0c, x0[:, 1:], xt[0,:], pt, model)
    else:
        pf = (x0 - np.mean(x0,axis=1).reshape(-1,1)) @ (x0 - np.mean(x0,axis=1).reshape(-1,1)).T/(nmem-1)
        #plot_initial(None, x0, xt[0,:], pt, model)
    logger.info("p0 max {}, min {}".format(np.max(pf),np.min(pf)))
    #print("pf={}".format(pf))
    u = np.zeros((nx, nmem))
    u[:, :] = x0
    #u[:, 0] = x0c
    #u[:, 1:] = x0
    xa = np.zeros((na, nx, nmem))
    xf = np.zeros_like(xa)
    xf[0, :, :] = u
    if pt == "mlef" or pt == "grad" or pt == "mlef05" or pt == "grad05" or pt == "mlefw" or pt == "mlef3":
        sqrtpa = np.zeros((na, nx, nmem-1))
    else:
        sqrtpa = np.zeros((na, nx, nx))
    e = np.zeros(na)
    chi = np.zeros(na)
    for i in range(na):
        y = obs[i]
        logger.debug("obs={}".format(y))
        logger.info("cycle{} analysis".format(i))
        #if i in range(0,4):
        #    u, pa, chi2, ds = analysis(u, y, rmat, rinv, sigma[op], htype, \
        #        method=method, cgtype=cgtype, \
        #        maxiter=maxiter, restart=restart, maxrest=maxrest, \
        #        hist=True, dh=True, \
        #        infl=linf, loc=lloc, tlm=ltlm, infl_parm = infl_parm, \
        #        model=model, icycle=i)
        #else:
        u, pa, chi2, ds = analysis(u, y, rmat, rinv, sigma[op], htype, \
                method=method, cgtype=cgtype, \
                maxiter=maxiter, restart=restart, maxrest=maxrest, \
                infl=linf, loc=lloc, tlm=ltlm, infl_parm = infl_parm, \
                model=model, icycle=i)
        xa[i, :, :] = u
        sqrtpa[i, :, :] = pa
        chi[i] = chi2
        if i < na-1:
            u = forecast(u, dt, F, nt, htype)
            xf[i+1, :, :] = u
        if pt == "mlef" or pt == "grad" or pt == "mlef05" or pt == "grad05" or pt == "mlefw" or pt == "mlef3":
            e[i] = np.sqrt(np.mean((xa[i, :, 0] - xt[i, :])**2))
        else:
            e[i] = np.sqrt(np.mean((np.mean(xa[i, :, :], axis=1) - xt[i, :])**2))
    #np.save("{}_ua_{}_{}.npy".format(model, op, pt), xa)
    #np.save("{}_uf_{}_{}.npy".format(model, op, pt), xf)
    #np.save("{}_pa_{}_{}.npy".format(model, op, pt), sqrtpa)
    if len(sys.argv) > 7:
        np.savetxt("{}_e_{}_{}_nt{}.txt".format(model, op, pt, nt), e)
        #np.savetxt("{}_chi_{}_{}_nt{}.txt".format(model, op, pt, nt), chi)
    #    np.savetxt("{}_e_{}_{}_{}.txt".format(model, op, pt, method_short), e)
    #    np.savetxt("{}_chi_{}_{}_{}.txt".format(model, op, pt, method_short), chi)
    else:
        np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
        np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
