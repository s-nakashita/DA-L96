import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
import obs
import costJ
from minimize import Minimize

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('anl')


def precondition(zmat):
    #u, s, vt = la.svd(zmat)
    #v = vt.transpose()
    #is2r = 1 / (1 + s**2)
    #tmat = v @ np.diag(np.sqrt(is2r)) @ vt
    #heinv = v @ np.diag(is2r) @ vt
    c = zmat.T @ zmat
    lam, v = la.eigh(c)
    D = np.diag(1.0/(np.sqrt(lam + np.ones(lam.size))))
    tmat = v @ D @ v.T
    heinv = tmat @ tmat.T
    logger.debug("tmat={}".format(tmat))
    logger.debug("heinv={}".format(heinv))
#    logger.debug("s={}".format(s))
#    print("s={}".format(s))
    return tmat, heinv


zetak = []
def callback(xk):
    global zetak
#    logger.debug("xk={}".format(xk))
    zetak.append(xk)


def calc_j(zeta, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
    x = xc + gmat @ zeta
    ob = y - obs.h_operator(x, htype["operator"])
    j = 0.5 * (zeta.transpose() @ heinv @ zeta + ob.transpose() @ rinv @ ob)
#    logger.debug("zeta.shape={}".format(zeta.shape))
#    logger.debug("j={} zeta={}".format(j, zeta))
    return j
    

def calc_grad_j(zeta, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
    x = xc + gmat @ zeta
    hx = obs.h_operator(x, htype["operator"])
    ob = y - hx
    if htype["perturbation"] == "grad05":
        dh = obs.dhdx(x, htype["operator"]) @ pf
    else:
        dh = obs.h_operator(x[:, None] + pf, htype["operator"]) - hx[:, None]
    return heinv @ zeta - tmat @ dh.transpose() @ rinv @ ob


def analysis(xf, xc, y, rmat, rinv, htype, gtol=1e-6, method="LBFGS",
        maxiter=None, disp=False, save_hist=False, save_dh=False, 
        model="z08", icycle=0):
    global zetak
    op = htype["operator"]
    pt = htype["perturbation"]
    pf = xf - xc[:, None]
#    logger.debug("norm(pf)={}".format(la.norm(pf)))
    if pt == "grad05":
#        logger.debug("dhdx.shape={}".format(obs.dhdx(xc, op).shape))
        dh = obs.dhdx(xc, op) @ pf
    else:
        dh = obs.h_operator(xf, op) - obs.h_operator(xc, op)[:, None]
    if save_dh:
        np.save("{}_dh_{}_{}.npy".format(model, op, pt), dh)
    logger.info("save_dh={}".format(save_dh))
#    print("save_dh={}".format(save_dh))
    zmat = rmat @ dh
#    logger.debug("cond(zmat)={}".format(la.cond(zmat)))
    tmat, heinv = precondition(zmat)
#    logger.debug("pf.shape={}".format(pf.shape))
#    logger.debug("tmat.shape={}".format(tmat.shape))
#    logger.debug("heinv.shape={}".format(heinv.shape))
    gmat = pf @ tmat
#    logger.debug("gmat.shape={}".format(gmat.shape))
    x0 = np.zeros(xf.shape[1])
    args_j = (xc, pf, y, tmat, gmat, heinv, rinv, htype)
    iprint = np.zeros(2, dtype=np.int32)
    minimize = Minimize(x0.size, 7, calc_j, calc_grad_j, 
                        args_j, iprint, method)
    logger.info("save_hist={}".format(save_hist))
#    print("save_hist={}".format(save_hist))
    cg = spo.check_grad(calc_j, calc_grad_j, x0, *args_j)
    logger.info("check_grad={}".format(cg))
    if save_hist:
        x = minimize(x0, callback=callback)
        logger.debug(zetak)
        jh = np.zeros(len(zetak))
        gh = np.zeros(len(zetak))
        for i in range(len(zetak)):
            jh[i] = calc_j(np.array(zetak[i]), *args_j)
            g = calc_grad_j(np.array(zetak[i]), *args_j)
            gh[i] = np.sqrt(g.transpose() @ g)
        np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), jh)
        np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), gh)
        if model=="z08":
            xmax = max(np.abs(np.min(x)),np.max(x))
            logger.debug("resx max={}".format(xmax))
            xmax = np.ceil(xmax*0.001)*1000
            logger.debug("resx max={}".format(xmax))
            costJ.cost_j2d(xmax, xf.shape[1], model, x, icycle,
                           htype, calc_j, calc_grad_j, *args_j)
        elif model=="l96":
            cost_j(200, xf.shape[1], model, x, icycle, *args_j)
    else:
        x = minimize(x0)
    xa = xc + gmat @ x
    if pt == "grad05":
        dh = obs.dhdx(xa, op) @ pf
    else:
        dh = obs.h_operator(xa[:, None] + pf, op) - obs.h_operator(xa, op)[:, None]
    zmat = rmat @ dh
#    logger.debug("cond(zmat)={}".format(la.cond(zmat)))
    tmat, heinv = precondition(zmat)
    d = y - obs.h_operator(xa, op)
    chi2 = chi2_test(zmat, heinv, rmat, d)
    pa = pf @ tmat 
    return xa, pa, chi2

def cost_j(nx, nmem, model, xopt, icycle, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
    op = htype["operator"]
    pt = htype["perturbation"]
    delta = np.linspace(-2000,2000,nx)
    jval = np.zeros((len(delta)+1,nmem))
    jval[0,:] = xopt
    for k in range(nmem):
        x0 = np.zeros(nmem)
        for i in range(len(delta)):
            x0[k] = delta[i]
            j = calc_j(x0, *args)
            jval[i+1,k] = j
    np.save("{}_cJ_{}_{}_cycle{}.npy".format(model, op, pt, icycle), jval)

def chi2_test(zmat, heinv, rmat, d):
    p = d.size
    G_inv = np.eye(p) - zmat @ heinv @ zmat.T
    innv = rmat @ d[:,None]
    return innv.T @ G_inv @ innv / p