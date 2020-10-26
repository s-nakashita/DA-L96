import sys
import logging
import numpy as np
from burgers import step
from obs import add_noise, h_operator
import mlef
#import mlef05 as mlef
import enkf
import matplotlib.pyplot as plt

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
t0off =  10 # initial offset between adjacent members
t0true = 20 # t0 for true
t0c =    60 # t0 for control
#t0c = t0true
            # t0 for ensemble members
nt =     20 # number of step per forecast
na =     20 # number of analysis

sigma = {"linear": 8.0e-2, "quadratic": 8.0e-2, "cubic": 7.0e-4, \
    "quadratic-nodiff": 8.0e-2, "cubic-nodiff": 7.0e-4,}
htype = {"operator": "linear", "perturbation": "mlef"}
linf = False
lloc = False
ltlm = True
obs_s = sigma[htype["operator"]]
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
if len(sys.argv) > 6:
    #t0off = int(sys.argv[6])
    obs_s = float(sys.argv[6])
t0m = [t0c + t0off//2 + t0off * i for i in range(-nmem//2, nmem//2)]
t0f = [t0c] + t0m
#logger.info("nmem={} t0true={} t0f={}".format(nmem, t0true, t0f))
print("nmem={} t0true={} t0f={}".format(nmem, t0true, t0f))
#logger.info("nt={} na={}".format(nt, na))
print("nt={} na={}".format(nt, na))
#logger.info("htype={} sigma={}".format(htype, sigma[htype["operator"]]))
print("htype={} sigma={}".format(htype, obs_s))
print("inflation={} localization={} TLM={}".format(linf,lloc,ltlm))

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
    y = h_operator(add_noise(u, sigma), op)
    return y


def forecast(u, dx, dt, nu, kmax, htype):
    for k in range(kmax):
        u = step(u, dx, dt, nu)
    if htype["perturbation"] == "etkf" or htype["perturbation"] == "po" \
        or ["perturbation"] == "letkf" or htype["perturbation"] == "srf":
        u[:, 0] = np.mean(u[:, 1:], axis=1)
    return u


def analysis(u, y, rmat, rinv, sig, htype, hist=False, dh=False,\
     infl=False, loc=False, tlm=True, \
     model="z08", icycle=0):
    #logger.info("hist={}".format(hist))
    print("hist={}".format(hist))
    if htype["perturbation"] == "mlef" or htype["perturbation"] == "grad":
        ua, pa, chi2= mlef.analysis(u[:, 1:], u[:, 0], y, rmat, rinv, htype,
                save_hist=hist, save_dh=dh, model=model, icycle=icycle)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
    else:
        u_ = np.mean(u[:,1:],axis=1)
        ua, ua_, pa, chi2 = enkf.analysis(u[:, 1:], u_, y, sig, dx, htype,\
             infl=infl, loc=loc, tlm=tlm, model=model)
        u[:, 0] = ua_
        u[:, 1:] = ua
    return u, pa, chi2

def plot_initial(u, ut, lag, model):
    fig, ax = plt.subplots()
    x = np.arange(ut.size) + 1
    ax.plot(x, ut, label="true")
    for i in range(u.shape[1]):
        if i==0:
            ax.plot(x, u[:,i], label="control")
        else:
            ax.plot(x, u[:,i], linestyle="--", label="mem{}".format(i+1))
    ax.set(xlabel="points", ylabel="u", title="initial lag={}".format(lag))
    ax.set_xticks(x[::5])
    ax.set_xticks(x, minor=True)
    ax.legend()
    fig.savefig("{}_initial_lag{}.png".format(model, lag))

if __name__ == "__main__":
    op = htype["operator"]
    pt = htype["perturbation"]
    model = "z08"
    rmat, rinv = set_r(nx, obs_s)
    ut, u = gen_true(x, dt, nu, t0true, t0f, nt, na)
    plot_initial(u, ut[0], t0off, model)
    #if pt == "mlef" or pt == "grad":
    #    p0 = u[:, 1:] - u[:, 0].reshape(-1,1) / np.sqrt(nmem-1)
    #    u[:, 1:] = u[:, 0].reshape(-1,1) + p0
    ua = np.zeros((na, nx, nmem+1))
    uf = np.zeros_like(ua)
    if pt == "mlef" or pt == "grad":
        sqrtpa = np.zeros((na, nx, nmem))
    else:
        sqrtpa = np.zeros((na, nx, nx))
    e = np.zeros(na)
    chi = np.zeros(na)
    for i in range(na):
        y = gen_obs(ut[i,], obs_s, op)
        if i in [0, 1, 2, 3]:
            #logger.info("first analysis")
            print("cycle{} analysis".format(i))
            u, pa, chi2 = analysis(u, y, rmat, rinv, obs_s, htype, \
                hist=True, dh=True,\
                infl=linf, loc=lloc, tlm=ltlm,\
                model=model, icycle=i)
        else:
            u, pa, chi2 = analysis(u, y, rmat, rinv, obs_s, htype, \
                infl=linf, loc=lloc, tlm=ltlm, model=model, icycle=i)
        ua[i, :, :] = u
        uf[i, :, :] = u
        sqrtpa[i,:,:] = pa
        chi[i] = chi2
        if i < na-1:
            u = forecast(u, dx, dt, nu, nt, htype)
        e[i] = np.sqrt(np.mean((ua[i, :, 0] - ut[i, :])**2))
    np.save("ut.npy", ut)
    np.save("{}_uf_{}_{}.npy".format(model, op, pt), uf)
    np.save("{}_ua_{}_{}.npy".format(model, op, pt), ua)
    np.save("{}_pa_{}_{}.npy".format(model, op, pt), sqrtpa)
    if len(sys.argv) > 6:
        oberr = -int(np.log10(obs_s))
        np.savetxt("{}_e_{}_{}_oberr{}.txt".format(model, op, pt, oberr), e)
        np.savetxt("{}_chi_{}_{}_oberr{}.txt".format(model, op, pt, oberr), chi)
    else:
        np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
        np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
