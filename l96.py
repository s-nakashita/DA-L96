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
dt = 0.05 / 6  # time step (=1 hour)
#logger.info("nx={} nu={} dt={:7.3e}".format(nx, nu, dt))
print("nx={} F={} dt={:7.3e}".format(nx, F, dt))

x = np.linspace(-2.0, 2.0, nx)
dx = x[1] - x[0]
np.savetxt("x.txt", x)

nmem =   40 # ensemble size
t0off =   8 # initial offset between adjacent members
t0c =    500 # t0 for control
            # t0 for ensemble members
t0m = [t0c + t0off//2 + t0off * i for i in range(-nmem//2, nmem//2)]
t0f = [t0c] + t0m
nt =     6 # number of step per forecast (=6 hour)
na =   100 # number of analysis
namax = 1460 # max number of analysis (1 year)

sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0}
#sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
#    "quadratic-nodiff": 1.0, "cubic-nodiff": 1.0, "test":1.0}
htype = {"operator": "linear", "perturbation": "mlef"}
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
    if sys.argv[4] == "T":
        linf = True
if len(sys.argv) > 5:
    if sys.argv[5] == "T":
        lloc = True
if len(sys.argv) > 6:
    if sys.argv[6] == "F":
        ltlm = False
#logger.info("nmem={} t0true={} t0f={}".format(nmem, t0true, t0f))
print("nmem={} t0f={}".format(nmem, t0f))
#logger.info("nt={} na={}".format(nt, na))
print("nt={} na={}".format(nt, na))
#logger.info("htype={} sigma={}".format(htype, sigma[htype["operator"]]))
print("htype={} sigma={}".format(htype, sigma[htype["operator"]]))
print("inflation={} localization={} TLM={}".format(linf,lloc,ltlm))

def set_r(nx, sigma):
    rmat = np.diag(np.ones(nx) / sigma)
    rinv = rmat.transpose() @ rmat
    return rmat, rinv

def get_true_and_obs(na, nx, sigma, op):
    truth = pd.read_csv("data.csv")
    xt = truth.values.reshape(na,nx)

    #obs = pd.read_csv("observation_data.csv")
    #y = obs.values.reshape(na,nx)
    y = add_noise(h_operator(xt, op), sigma)

    return xt, y

def init_ens(nx,nmem,t0c,t0f,dt,F,opt):
    X0c = np.ones(nx)*F
    X0c[nx//2 - 1] += 0.001*F
    tmp = np.zeros_like(X0c)
    maxiter = np.max(np.array(t0f))+1
    if(opt==0): # random
        print("spin up max = {}".format(t0c))
        np.random.seed(514)
        for j in range(t0c):
            #X0 = step(X0, dt, F)
            X0c = step(X0c, dt, F)
        X0 = np.random.normal(0.0,1.0,size=(nx,nmem)) + X0c[:, None]
        #for j in range(t0c):
        #    X0 = step(X0, dt, F)
        #    X0c = step(X0c, dt, F)
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


def analysis(u, y, rmat, rinv, sig, htype, hist=False, dh=False, \
    infl=False, loc=False, tlm=True,\
    model="l96", icycle=0):
    #logger.info("hist={}".format(hist))
    print("hist={}".format(hist))
    if htype["perturbation"] == "mlef" or htype["perturbation"] == "grad":
        ua, pa, chi2, ds= mlef.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype, \
            save_hist=hist, save_dh=dh, \
            infl = infl, loc = loc, model=model, icycle=icycle)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
    else:
        u_ = np.mean(u[:,1:],axis=1)
        ua, ua_, pa, chi2, ds = enkf.analysis(u[:, 1:], u_, y, sig, dx, htype, \
            infl = infl, loc = loc, tlm=tlm, \
            save_dh=dh, model=model, icycle=icycle)
        u[:, 0] = ua_
        u[:, 1:] = ua
    return u, pa, chi2, ds

def plot_initial(uc, u, ut, lag, model):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = np.arange(ut.size) + 1
    ax.plot(x, ut, label="true")
    ax.plot(x, uc, label="control")
    for i in range(u.shape[1]):
        ax.plot(x, u[:,i], linestyle="--", label="mem{}".format(i+1))
    ax.set(xlabel="points", ylabel="X", title="initial lag={}".format(lag))
    ax.set_xticks(x[::5])
    ax.set_xticks(x, minor=True)
    ax.legend()
    fig.savefig("{}_initial_lag{}.png".format(model, lag))

if __name__ == "__main__":
    op = htype["operator"]
    pt = htype["perturbation"]
    model = "l96"
    rmat, rinv = set_r(nx, sigma[op])
    xt, obs = get_true_and_obs(namax, nx, sigma[op], op)
    np.save("{}_ut.npy".format(model), xt[:na,:])
    x0c, x0 = init_ens(nx,nmem,t0c,t0f,dt,F,opt=0)
    if pt == "mlef" or pt == "grad":
        pf = (x0 - x0c[:,None]) @ (x0 - x0c[:,None]).T
    else:
        pf = (x0 - np.mean(x0,axis=1).reshape(-1,1)) @ (x0 - np.mean(x0,axis=1).reshape(-1,1)).T/(nmem-1)
    print("pf={}".format(pf))
    u = np.zeros((nx, nmem+1))
    u[:, 0] = x0c
    u[:, 1:] = x0
    xa = np.zeros((na, nx, nmem+1))
    xf = np.zeros_like(xa)
    if pt == "mlef" or pt == "grad":
        sqrtpa = np.zeros((na, nx, nmem))
    else:
        sqrtpa = np.zeros((na, nx, nx))
    e = np.zeros(na)
    chi = np.zeros(na)
    for i in range(na):
        y = obs[i]
        if i in range(0,4):
            #logger.info("first analysis")
            print("cycle{} analysis".format(i))
            u, pa, chi2, ds = analysis(u, y, rmat, rinv, sigma[op], htype, \
                hist=True, dh=True, \
                infl=linf, loc=lloc, tlm=ltlm,\
                model=model, icycle=i)
        else:
            u, pa, chi2, ds = analysis(u, y, rmat, rinv, sigma[op], htype, \
                infl=linf, loc=lloc, tlm=ltlm, \
                model=model, icycle=i)
        xa[i, :, :] = u
        xf[i, :, :] = u
        sqrtpa[i, :, :] = pa
        chi[i] = chi2
        if i < na-1:
            u = forecast(u, dt, F, nt, htype)
        e[i] = np.sqrt(np.mean((xa[i, :, 0] - xt[i, :])**2))
    np.save("{}_ua_{}_{}.npy".format(model, op, pt), xa)
    np.save("{}_pa_{}_{}.npy".format(model, op, pt), sqrtpa)
    if len(sys.argv) > 7:
        np.savetxt("{}_e_{}_{}_lag{}.txt".format(model, op, pt, t0off), e)
        np.savetxt("{}_chi_{}_{}_lag{}.txt".format(model, op, pt,t0off), chi)
    else:
        np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
        np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
