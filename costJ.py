import sys
import logging
import numpy as np
from burgers import step
import obs


#logging.config.fileConfig("logging_config.ini")
#logger = logging.getLogger()

nx = 81     # number of points
nu = 0.05   # diffusion
dt = 0.0125 # time step
#logger.info("nx={} nu={} dt={:7.3e}".format(nx, nu, dt))
#print("nx={} nu={} dt={:7.3e}".format(nx, nu, dt))

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
#print("nmem={} t0true={} t0f={}".format(nmem, t0true, t0f))
#logger.info("nt={} na={}".format(nt, na))
#print("nt={} na={}".format(nt, na))

sigma = {"linear": 8.0e-2, "quadratic": 1.0e-3, "cubic": 1.0e-3, "quartic": 1.0e-3, \
    "quadratic-nodiff": 1.0e-3, "cubic-nodiff": 1.0e-3, "quartic-nodiff": 1.0e-3}
htype = {"operator": "linear", "perturbation": "mlef"}
if len(sys.argv) > 1:
    htype["operator"] = sys.argv[1]
if len(sys.argv) > 2:
    htype["perturbation"] = sys.argv[2]
#logger.info("htype={} sigma={}".format(htype, sigma[htype["operator"]]))
#print("htype={} sigma={}".format(htype, sigma[htype["operator"]]))


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
    y = obs.add_noise(obs.h_operator(u, sigma), op)
    return y

def precondition(zmat):
    u, s, vt = la.svd(zmat)
    v = vt.transpose()
    is2r = 1 / (1 + s**2)
    tmat = v @ np.diag(np.sqrt(is2r)) @ vt
    heinv = v @ np.diag(is2r) @ vt
#    logger.debug("tmat={}".format(tmat))
#    logger.debug("heinv={}".format(heinv))
#    logger.debug("s={}".format(s))
    print("s={}".format(s))
    return tmat, heinv

def calc_jb(zeta, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
    jb = 0.5 * zeta.transpose() @ heinv @ zeta
    #j = 0.5 * ((nmem-1)*zeta.transpose() @ heinv @ zeta + nob.transpose() @ rinv @ ob)
#    logger.debug("zeta.shape={}".format(zeta.shape))
#    logger.debug("j={} zeta={}".format(j, zeta))
    return jb

def calc_jo(zeta, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
    nmem = zeta.size
    x = xc + gmat @ zeta
    ob = y - obs.h_operator(x, htype["operator"])
    jo = 0.5 * ob.transpose() @ rinv @ ob
    #j = 0.5 * ((nmem-1)*zeta.transpose() @ heinv @ zeta + nob.transpose() @ rinv @ ob)
#    logger.debug("zeta.shape={}".format(zeta.shape))
#    logger.debug("j={} zeta={}".format(j, zeta))
    return jo

def cost_j(nx, nmem, *args_j):
    delta = np.linspace(-nx,nx,4*nx)
    jvalb = np.zeros((len(delta),nmem))
    jvalo = np.zeros((len(delta),nmem))
    for k in range(nmem):
        x0 = np.zeros(nmem)
        for i in range(len(delta)):
            x0[k] = delta[i]
            jb = calc_jb(x0, *args_j)
            jo = calc_jo(x0, *args_j)
            jvalb[i,k] = jb
            jvalo[i,k] = jo
    return jvalb, jvalo
    #np.save("cJ_{}_{}.npy".format(op, pt), jval)

def cost_j2d(xmax, nmem, model, xopt, icycle, htype, calc_j, calc_grad_j, *args):
    op = htype["operator"]
    pt = htype["perturbation"]
    delta = np.linspace(-xmax,xmax,101, endpoint=True)
    x_g = np.zeros((3,nmem))
    x_g[0,:] = xopt
    x_g[1,:] = calc_grad_j(np.zeros(nmem), *args)
    x_g[2,0] = xmax
    np.save("{}_x+g_{}_{}_cycle{}.npy".format(model, op, pt, icycle), x_g)

    for i in range(nmem-1):
        for j in range(i+1,nmem):
            jval = np.zeros((len(delta),len(delta)))
            x0 = np.zeros(nmem)
            for k in range(len(delta)):
                x0[i] = delta[k]
                for l in range(len(delta)):
                    x0[j] = delta[l]
                    jval[l,k] = calc_j(x0, *args)
            np.save("{}_cJ2d_{}_{}_{}{}_cycle{}.npy".format(model, op, pt, i, j, icycle), jval)
if __name__ == "__main__":
    op = htype["operator"]
    pt = htype["perturbation"]
    rmat, rinv = set_r(nx, sigma[op])
    ut, u = gen_true(x, dt, nu, t0true, t0f, nt, na)
    xc = u[:,0]
    xf = u[:,1:]
    pf = xf - xc[:,None]
    dh = obs.h_operator(xf, op) - obs.h_operator(xc, op)[:,None]
    zmat = rmat @ dh
    tmat, heinv = precondition(zmat)
    gmat = pf @ tmat
    y = gen_obs(ut[0,], sigma[op], op)

    args_j = (xc, pf, y, tmat, gmat, heinv, rinv, htype)
    cost_j(1000, xf.shape[1], *args_j)
    np.save("cJ_{}_{}.npy".format(op, pt), jval)
