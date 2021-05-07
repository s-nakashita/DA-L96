import sys
import logging
from logging.config import fileConfig
import os
import numpy as np
from burgers import step
from obs import add_noise, h_operator
import mlef
import mlefb
import mleft
import mlef05
import mlef08
import mlefsc
import mlef2 as mlefw
import mlef3
import enkf
import matplotlib.pyplot as plt

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger()

global nx, nu, dt, dx

nx = 81     # number of points
nu = 0.05   # diffusion
dt = 0.0125 # time step
logger.info("nx={} nu={} dt={:7.3e}".format(nx, nu, dt))
#print("nx={} nu={} dt={:7.3e}".format(nx, nu, dt))

x = np.linspace(-2.0, 2.0, nx)
dx = x[1] - x[0]
np.savetxt("x.txt", x)

nmem =    4 # ensemble size (not include control)
t0off =  12 # initial offset between adjacent members
t0true = 20 # t0 for true
t0c =    60 # t0 for control
#t0c = t0true
t0l = t0c + t0off // 2 - t0off * nmem//2
t0r = t0c + t0off // 2 + t0off * nmem//2
            # t0 for ensemble members
nt =     20 # number of step per forecast
na =     20 # number of analysis

#sigma = {"linear": 8.0e-2, "quadratic": 8.0e-2, "cubic": 7.0e-4, "quartic": 7.0e-4, \
#        "quadratic-nodiff": 8.0e-2, "cubic-nodiff": 7.0e-4, "quartic-nodiff": 7.0e-4}
sigma = {"linear": 1.0e-3, "quadratic": 1.0e-3, "cubic": 1.0e-3, "quartic": 1.0e-3, \
    "quadratic-nodiff": 1.0e-3, "cubic-nodiff": 1.0e-3, "quartic-nodiff": 1.0e-3, "test":8.0e-2}
htype = {"operator": "linear", "perturbation": "mlef", "gamma": 1}
linf = False
lloc = False
ltlm = True
if len(sys.argv) > 1:
    htype["operator"] = sys.argv[1]
if len(sys.argv) > 2:
    htype["perturbation"] = sys.argv[2]
if len(sys.argv) > 3:
    if sys.argv[3] == "T":
        linf = True
if len(sys.argv) > 4:
    if sys.argv[4] == "T":
        lloc = True
if len(sys.argv) > 5:
    if sys.argv[5] == "F":
        ltlm = False
obs_s = sigma[htype["operator"]]
if len(sys.argv) > 6:
    #nmem = int(sys.argv[6])
    #t0off = min(t0off, 2 * t0c // nmem)
    obs_s = float(sys.argv[6])
maxiter = None # inner-loop iteration
#if len(sys.argv) > 6:
#    #htype["gamma"] = int(sys.argv[7])
#    maxiter = int(sys.argv[6])
if len(htype["operator"]) > 11: # non-differentiable
    method = "LBFGS"
    method_short = "lb"
    cgtype = None
else:
    method = "CGF"
    method_short = "cgf_prb"
    cgtype = 3
#method = "CG"
#method_short = "cg"
#cgtype = None
if len(sys.argv) > 7:
    method_short = sys.argv[7]
    if method_short == "lb":
        method = "LBFGS"
    elif method_short == "bg":
        method = "BFGS"
    elif method_short == "cg":
        method = "CG"
    elif method_short == "nm":
        method = "Nelder-Mead"
    elif method_short == "pw":
        method = "Powell"
    elif method_short == "gd":
        method = "GD"
    elif method_short == "gdf":
        method = "GDF"
    elif method_short == "ncg":
        method = "NCG"
    elif method_short == "tnc":
        method = "TNC"
    elif method_short == "trn":
        method = "trust-ncg"
    elif method_short == "trk":
        method = "trust-krylov"
    elif method_short == "tre":
        method = "trust-exact"
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
if len(sys.argv) > 8:
    if sys.argv[8] == "T":
        restart = True
    elif sys.argv[8] == "F":
        restart = False
# initial lag
lplot = False
if len(sys.argv) > 9:
    t0c = int(sys.argv[9])
    if sys.argv[10] == "T":
        lplot = True
t0m = [t0c + t0off//2 + t0off * i for i in range(-nmem//2, nmem//2)]
t0f = [t0c] + t0m
logger.info("nmem={} t0true={} t0f={}".format(nmem, t0true, t0f))
#print("nmem={} t0true={} t0f={}".format(nmem, t0true, t0f))
logger.info("nt={} na={}".format(nt, na))
#print("nt={} na={}".format(nt, na))
logger.info("htype={} sigma={}".format(htype, obs_s))
#print("htype={} sigma={}".format(htype, obs_s))
logger.info("inflation={} localization={} TLM={}".format(linf,lloc,ltlm))
#print("inflation={} localization={} TLM={}".format(linf,lloc,ltlm))
logger.info("maximum minimization iteration={}".format(maxiter))
#print("maximum minimization iteration={}".format(maxiter))
logger.info("minimization method={} restart={}".format(method, restart))

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
        if k+1 in t0f:
            u0[:, t0f.index(k+1)] = u
    ut[0, :] = u
    for i in range(na-1):
        for k in range(nt):
            u = step(u, dx, dt, nu)
            #j = (i + 1) * nt + k
            j = t0true + i * nt + k
            if j in t0f:
                u0[:, t0f.index(j)] = u
        ut[i+1, :] = u
    return ut, u0


def gen_obs(u, sigma, op):
    seed = None
    y = add_noise(h_operator(u, op), sigma, seed=seed)
    return y

def get_obs(f):
    y = np.load(f)
    return y


def forecast(u, dx, dt, nu, kmax, htype):
    for k in range(kmax):
        u = step(u, dx, dt, nu)
    #if htype["perturbation"] == "etkf" or htype["perturbation"] == "po" \
    #    or htype["perturbation"] == "letkf" or htype["perturbation"] == "srf" \
    #    or htype["perturbation"] == "mlefb" or htype["perturbation"] == "gradb":
    #    u[:, 0] = np.mean(u[:, 1:], axis=1)
    return u


def analysis(u, y, rmat, rinv, sig, htype, hist=False, dh=False,\
     method="LBFGS", cgtype=None,\
     maxiter=None, restart=True, maxrest=20, \
     infl=False, loc=False, tlm=True, \
     model="z08", icycle=0):
    from copy import deepcopy
    logger.info("hist={}".format(hist))
    logger.info("cycle={}".format(icycle))
    #print("hist={}".format(hist))
    if htype["perturbation"] == "mlef" or htype["perturbation"] == "grad":
        ua, pa, chi2, ds, condh = mlef.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype,
            method=method, maxiter=maxiter, save_hist=hist, save_dh=dh, model=model, icycle=icycle)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
    elif htype["perturbation"] == "mlef08" or htype["perturbation"] == "grad08":
        ua, pa, chi2, ds, condh = mlef08.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype,
            method=method, maxiter=maxiter, save_hist=hist, save_dh=dh, model=model, icycle=icycle)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
    elif htype["perturbation"] == "mlef05" or htype["perturbation"] == "grad05":
        ua, pa, chi2 = mlef05.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype,
            method=method, cgtype=cgtype, 
            maxiter=maxiter, restart=restart, maxrest=maxrest,
            save_hist=hist, save_dh=dh, model=model, icycle=icycle)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
        ds = 0.0
        condh = 0.0
    elif htype["perturbation"] == "mlefsc":
        ua, pa, chi2 = mlefsc.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype,
            method=method, cgtype=cgtype, maxiter=maxiter, save_hist=hist, save_dh=dh, model=model, icycle=icycle)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
        ds = 0.0
        condh = 0.0
    elif htype["perturbation"] == "mlefw" or htype["perturbation"] == "gradw":
        ua, pa, chi2 = mlefw.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype,
            method=method, maxiter=None, disp=True,
            save_hist=hist, save_dh=dh, model=model, icycle=icycle)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
        ds = 0.0
        condh = 0.0
    elif htype["perturbation"] == "mlefh":
        if icycle > 0:
            ua, pa, chi2, ds, condh = mlef.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype, 
            method=method, maxiter=maxiter, save_hist=hist, save_dh=dh, model=model, icycle=icycle)
            u[:, 0] = ua 
            u[:, 1:] = ua[:, None] + pa 
        else:
            ua, pa, chi2 = mlefw.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype,
            method=method, maxiter=maxiter, save_hist=hist, save_dh=dh, model=model, icycle=icycle)
            u[:, 0] = ua
            u[:, 1:] = ua[:, None] + pa
            ds = 0.0
            condh = 0.0
    elif htype["perturbation"] == "mlef3":
        ua, pa, chi2, ds = mlef3.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype,
            method=method, maxiter=maxiter, save_hist=hist, save_dh=dh, model=model, icycle=icycle)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
        condh = 0.0
    elif htype["perturbation"] == "mlefb" :
        u_ = np.mean(u,axis=1)
        ua, ua_, pa, chi2, ds, condh = mlefb.analysis(u, u_, y, rmat, rinv, htype,\
            method=method, maxiter=maxiter, save_hist=hist, save_dh=dh, model=model, icycle=icycle)
        u = ua
    elif htype["perturbation"] == "mleft" :
        u_ = np.mean(u,axis=1)
        ua, ua_, pa, chi2, ds, condh = mleft.analysis(u, u_, y, rmat, rinv, htype,\
            method=method, maxiter=maxiter, save_hist=hist, save_dh=dh, model=model, icycle=icycle)
        u = ua
    else:
        u_ = np.mean(u,axis=1)
        ua, ua_, pa, chi2, ds, condh = enkf.analysis(u, u_, y, sig, dx, htype,\
             save_dh=dh, infl=infl, loc=loc, tlm=tlm, model=model, icycle=icycle)
        u = ua
    return u, pa, chi2, ds, condh

def plot_initial(u, ut, t0c, model):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    x = np.arange(ut.size) + 1
    ax.plot(x, ut, label="true")
    for i in range(u.shape[1]):
        if i==0:
            ax.plot(x, u[:,i], label="control")
        else:
            ax.plot(x, u[:,i], linestyle="--", color="tab:green") #, label="mem{}".format(i))
    ax.set(xlabel="points", ylabel="u", title="initial lag={}".format(t0off))
    ax.set_xticks(x[::10])
    ax.set_xticks(x[::5], minor=True)
    ax.set_ylim([0.0,1.2])
    ax.legend()
    fig.savefig("{}_initial_t0c{}.png".format(model, t0c))
    #fig.savefig("{}_initial_nmem{}.pdf".format(model, u.shape[1]-1))

if __name__ == "__main__":
    op = htype["operator"]
    pt = htype["perturbation"]
    model = "z08"
    rmat, rinv = set_r(nx, obs_s)
    ut, u = gen_true(x, dt, nu, t0true, t0f, nt, na)
    oberr = int(obs_s*1e5)
    obsfile="obs_{}_{}.npy".format(op, oberr)
    logger.debug(obsfile)
    if not os.path.isfile(obsfile):
        logger.info("create obs")
#        print("create obs")
        obs = gen_obs(ut, obs_s, op)
        np.save(obsfile, obs)
    else:
        logger.info("read obs")
#        print("read obs")
        obs = get_obs(obsfile)
    if lplot:
        plot_initial(u, ut[0], t0c, model)
    #if pt == "mlef" or pt == "grad":
    #    p0 = u[:, 1:] - u[:, 0].reshape(-1,1) / np.sqrt(nmem-1)
    #    u[:, 1:] = u[:, 0].reshape(-1,1) + p0
    ua = np.zeros((na, nx, nmem+1))
    uf = np.zeros_like(ua)
    uf[0, :, :] = u
    #if pt == "mlef" or pt == "grad":
    #    sqrtpa = np.zeros((na, nx, nmem))
    #else:
    #    sqrtpa = np.zeros((na, nx, nx))
    e = np.zeros(na+1)
    if pt == "mlef" or pt == "grad" or pt == "mlef08" or pt == "grad08" or \
        pt == "mlef05" or pt == "grad05" or pt == "mlefsc" or pt == "mlefw" or pt == "gradw" or\
        pt == "mlef3" or pt == "mlefh":
        e[0] = np.sqrt(np.mean((uf[0, :, 0] - ut[0, :])**2))
    else:
        e[0] = np.sqrt(np.mean((np.mean(uf[0, :, :], axis=1) - ut[0, :])**2))
    dpf = np.zeros(na+1)
    if pt == "mlef" or pt == "grad" or pt == "mlef08" or pt == "grad08" or \
        pt == "mlef05" or pt == "grad05" or pt == "mlefsc" or pt == "mlefw" or pt == "gradw" or\
        pt == "mlef3" or pt == "mlefh":
        pf = u[:, 1:] - u[:, 0].reshape(-1,1)
        dpf[0] = np.sqrt(np.mean(np.diag(pf@pf.T)))
    else:
        pf = (u - np.mean(u, axis=1).reshape(-1,1))/np.sqrt(nmem)
        dpf[0] = np.sqrt(np.mean(np.diag(pf@pf.T)))
    chi = np.zeros(na)
    dof = np.zeros(na)
    dpa = np.zeros(na)
    ndpa = np.zeros(na)
    #if pt == "mlef" or pt == "mlefb":
    #    cond = np.zeros((na,2))
    #else:
    cond = np.zeros(na)
    #checkg = np.zeros(na)
    for i in range(na):
        #y = gen_obs(ut[i,], obs_s, op)
        y = obs[i]
        logger.info("cycle{} analysis".format(i))
        #if i in range(1):
        if i == -1:
            #logger.info("first analysis")
#            print("cycle{} analysis".format(i))
            u, pa, chi2, ds, condh = analysis(u, y, rmat, rinv, obs_s, htype, \
                method=method, cgtype=cgtype, \
                maxiter=maxiter, restart=restart,\
                hist=True, dh=True,\
                infl=linf, loc=lloc, tlm=ltlm,\
                model=model, icycle=i)
        else:
            u, pa, chi2, ds, condh = analysis(u, y, rmat, rinv, obs_s, htype, \
                method=method, cgtype=cgtype, \
                maxiter=maxiter, restart=restart,\
                infl=linf, loc=lloc, tlm=ltlm, \
                model=model, icycle=i)
        #print("ua={}".format(u))
        ua[i, :, :] = u
        #sqrtpa[i,:,:] = pa
        if pt == "mlef" or pt == "grad" or pt == "mlef08" or pt == "grad08" or \
        pt == "mlef05" or pt == "grad05" or pt == "mlefsc" or pt == "mlefw" or pt == "gradw" or\
        pt == "mlef3" or pt == "mlefh":
            dpa[i] = np.sqrt(np.mean(np.diag(pa@pa.T)))
            ndpa[i] = np.sqrt(np.mean(np.sum(pa@pa.T) - np.sum(np.diag(pa@pa.T))))
        else:
            dpa[i] = np.sqrt(np.mean(np.diag(pa)))
            ndpa[i] = np.sqrt(np.mean(np.sum(pa) - np.sum(np.diag(pa))))
        chi[i] = chi2
        dof[i] = ds
        #if pt == "mlef" or pt == "mlefb":
        #    cond[i,:] = condh
        #else:
        cond[i] = condh
        #checkg[i] = cg
        if i < na-1:
            u = forecast(u, dx, dt, nu, nt, htype)
            uf[i+1, :, :] = u
        #uf[i, :, :] = u
            if pt == "mlef" or pt == "grad" or pt == "mlef08" or pt == "grad08" or \
            pt == "mlef05" or pt == "grad05" or pt == "mlefsc" or pt == "mlefw" or pt == "gradw" or\
            pt == "mlef3" or pt == "mlefh":
                pf = u[:, 1:] - u[:, 0].reshape(-1,1)
                dpf[i+1] =  np.sqrt(np.mean(np.diag(pf@pf.T)))
            else:
                pf = (u - np.mean(u, axis=1).reshape(-1,1))/np.sqrt(nmem-1)
                dpf[i+1] =  np.sqrt(np.mean(np.diag(pf@pf.T)))
                #print("xf={}".format(u))
        if pt == "mlef" or pt == "grad" or pt == "mlef08" or pt == "grad08" or \
        pt == "mlef05" or pt == "grad05" or pt == "mlefsc" or pt == "mlefw" or pt == "gradw" or\
        pt == "mlef3" or pt == "mlefh":
            e[i+1] = np.sqrt(np.mean((ua[i, :, 0] - ut[i, :])**2))
        else:
            e[i+1] = np.sqrt(np.mean((np.mean(ua[i, :, :], axis=1) - ut[i, :])**2))
    np.save("{}_ut.npy".format(model), ut)
    np.save("{}_uf_{}_{}.npy".format(model, op, pt), uf)
    np.save("{}_ua_{}_{}.npy".format(model, op, pt), ua)
    ##np.save("{}_pa_{}_{}.npy".format(model, op, pt), sqrtpa)
    np.savetxt("{}_dpf_{}_{}.txt".format(model, op, pt), dpf)
    np.savetxt("{}_dpa_{}_{}.txt".format(model, op, pt), dpa)
    np.savetxt("{}_ndpa_{}_{}.txt".format(model, op, pt), ndpa)
    if len(sys.argv) > 7:
        oberr = str(int(obs_s*1e5)).zfill(5)
        #np.savetxt("{}_e_{}_{}_oberr{}_{}.txt".format(model, op, pt, oberr, method_short), e)
        np.savetxt("{}_e_{}_{}_oberr{}_lag{}.txt".format(model, op, pt, oberr, t0c), e)
        #np.savetxt("{}_e_{}_{}_nmem{}_{}.txt".format(model, op, pt, nmem, method_short), e)
        #np.savetxt("{}_chi_{}_{}_oberr{}_{}_{}.txt".format(model, op, pt, oberr, method_short), chi)
        #np.savetxt("{}_dof_{}_{}_oberr{}_{}_{}.txt".format(model, op, pt, oberr, method_short), dof)
    elif len(sys.argv) > 6:
    #if len(sys.argv) > 6:
        oberr = str(int(obs_s*1e5)).zfill(5)
        np.savetxt("{}_e_{}_{}_oberr{}.txt".format(model, op, pt, oberr), e)
        np.savetxt("{}_chi_{}_{}_oberr{}.txt".format(model, op, pt, oberr), chi)
        np.savetxt("{}_dof_{}_{}_oberr{}.txt".format(model, op, pt, oberr), dof)
        #np.savetxt("{}_e_{}_{}_{}.txt".format(model, op, pt, method_short), e)
        #np.savetxt("{}_chi_{}_{}_{}.txt".format(model, op, pt, method_short), chi)
        #np.savetxt("{}_dof_{}_{}_{}.txt".format(model, op, pt, method_short), dof)
    else:
        np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
        np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
        np.savetxt("{}_dof_{}_{}.txt".format(model, op, pt), dof)
    #np.savetxt("{}_condh_{}_{}.txt".format(model, op, pt), cond)
    #np.savetxt("{}_checkg_{}_{}.txt".format(model, op, pt), checkg)
