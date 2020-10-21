import sys
import logging
import numpy as np
from burgers import step
from obs import add_noise, h_operator
from mlef import calc_j, precondition


#logging.config.fileConfig("logging_config.ini")
#logger = logging.getLogger()

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
    "quadratic-nodiff": 8.0e-4, "cubic-nodiff": 7.0e-4,}
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

def cost_j(nx, nmem, *args_j):
    delta = np.linspace(-100,100,nx)
    jval = np.zeros((len(delta),nmem))
    for k in range(nmem):
        x0 = np.zeros(nmem)
        for i in range(len(delta)):
            x0[k] = delta[i]
            j = calc_j(x0, *args_j)
            jval[i,k] = j
    np.save("cJ_{}_{}.npy".format(op, pt), jval)

if __name__ == "__main__":
    op = htype["operator"]
    pt = htype["perturbation"]
    rmat, rinv = set_r(nx, sigma[op])
    ut, u = gen_true(x, dt, nu, t0true, t0f, nt, na)
    xc = u[:,0]
    xf = u[:,1:]
    pf = xf - xc[:,None]
    dh = h_operator(xf, op) - h_operator(xc, op)[:,None]
    zmat = rmat @ dh
    tmat, heinv = precondition(zmat)
    gmat = pf @ tmat
    y = gen_obs(ut[0,], sigma[op], op)

    args_j = (xc, pf, y, tmat, gmat, heinv, rinv, htype)
    cost_j(1000, xf.shape[1], *args_j)
