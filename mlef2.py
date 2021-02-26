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
    Dinv = np.diag((np.sqrt(lam + np.ones(lam.size))))
    tinv = v @ Dinv @ v.T
    heinv = tmat @ tmat.T
    logger.debug("tmat={}".format(tmat))
    logger.debug("heinv={}".format(heinv))
#    logger.debug("s={}".format(s))
    #print("s={}".format(s))
    return tmat, heinv


wk = []
def callback(xk):
    global wk
#    logger.debug("xk={}".format(xk))
    wk.append(xk)


def calc_j(w, *args):
    xc, pf, y, rinv, htype = args
    x = xc + pf @ w
    ob = y - obs.h_operator(x, htype["operator"])
    j = 0.5 * (w.transpose() @ w + ob.transpose() @ rinv @ ob)
#    logger.debug("zeta.shape={}".format(zeta.shape))
#    logger.debug("j={} zeta={}".format(j, zeta))
    return j
    

def calc_grad_j(w, *args):
    xc, pf, y, rinv, htype = args
    x = xc + pf @ w
    hx = obs.h_operator(x, htype["operator"])
    ob = y - hx
    if htype["perturbation"] == "grad":
        dh = obs.dhdx(x, htype["operator"]) @ pf
    else:
        dh = obs.h_operator(x[:, None] + pf, htype["operator"]) - hx[:, None]
    return w - dh.transpose() @ rinv @ ob


def analysis(xf, xc, y, rmat, rinv, htype, gtol=1e-6, method="LBFGS",
       maxiter=None, disp=False, save_hist=False, save_dh=False, 
       model="z08", icycle=0):
    global wk
    op = htype["operator"]
    pt = htype["perturbation"]
    nmem = xf.shape[1]
    pf = xf - xc[:, None]
    #pf = (xf - xc[:, None]) / np.sqrt(nmem)
#    logger.debug("norm(pf)={}".format(la.norm(pf)))
    if pt == "grad":
#        logger.debug("dhdx.shape={}".format(obs.dhdx(xc, op).shape))
        dh = obs.dhdx(xc, op) @ pf
    else:
        dh = obs.h_operator(xf, op) - obs.h_operator(xc, op)[:, None]
    if save_dh:
        np.save("{}_dh_{}_{}.npy".format(model, op, pt), dh)
    logger.info("save_dh={}".format(save_dh))
#    print("save_dh={}".format(save_dh))
    x0 = np.zeros(xf.shape[1])
    args_j = (xc, pf, y, rinv, htype)
    iprint = np.zeros(2, dtype=np.int32)
    minimize = Minimize(x0.size, 7, calc_j, calc_grad_j, 
                        args_j, iprint, method)
    logger.info("save_hist={}".format(save_hist))
    cg = spo.check_grad(calc_j, calc_grad_j, x0, *args_j)
    logger.info("check_grad={}".format(cg))
#    print("save_hist={}".format(save_hist))
    if save_hist:
        #g = calc_grad_j(x0, *args_j)
        #print("g={}".format(g))
        x = minimize(x0, callback=callback)
        logger.debug(wk)
        jh = np.zeros(len(wk))
        gh = np.zeros(len(wk))
        for i in range(len(wk)):
            jh[i] = calc_j(np.array(wk[i]), *args_j)
            g = calc_grad_j(np.array(wk[i]), *args_j)
            gh[i] = np.sqrt(g.transpose() @ g)
        np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), jh)
        np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), gh)
        if model=="z08":
            xmax = max(np.abs(np.min(x)),np.max(x))
            logger.debug("resx max={}".format(xmax))
#            print("resx max={}".format(xmax))
            xmax = np.ceil(xmax*0.001)*1000
            logger.debug("resx max={}".format(xmax))
#           print("resx max={}".format(xmax))
            #cost_j(xmax, xf.shape[1], model, x, icycle, *args_j)
            #cost_j2d(xmax, xf.shape[1], model, x, icycle, *args_j)
            costJ.cost_j2d(xmax, xf.shape[1], model, x, icycle,
                           htype, calc_j, calc_grad_j, *args_j)
        elif model=="l96":
            cost_j(200, xf.shape[1], model, x, icycle, *args_j)
    else:
        x = minimize(x0)
    xa = xc + pf @ x
    
    if pt == "grad":
        dh = obs.dhdx(xa, op) @ pf
    else:
        dh = obs.h_operator(xa[:, None] + pf, op) - obs.h_operator(xa, op)[:, None]
    zmat = rmat @ dh
#    logger.debug("cond(zmat)={}".format(la.cond(zmat)))
    tmat, heinv = precondition(zmat)
    d = y - obs.h_operator(xa, op)
    chi2 = chi2_test(zmat, heinv, rmat, d)
    pa = pf @ tmat #* np.sqrt(nmem)
    return xa, pa, chi2

def cost_j(nx, nmem, model, xopt, icycle, *args):
    xc, pf, y, rinv, htype = args
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