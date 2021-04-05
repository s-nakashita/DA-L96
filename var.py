import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
import obs
from minimize import Minimize

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')

zetak = []
alphak = []
def callback(xk, alpha=None):
    global zetak, alphak
    zetak.append(xk)
    if alpha is not None:
        alphak.append(alpha)

def calc_j(x, *args):
    binv, JH, rinv, ob = args
    jb = 0.5 * x.T @ binv @ x
    d = JH @ x - ob
    jo = 0.5 * d.T @ rinv @ d
    return jb + jo

def calc_grad_j(x, *args):
    binv, JH, rinv, ob = args
    d = JH @ x - ob
    return binv @ x + JH.T @ rinv @ d 

def calc_hess(x, *args):
    binv, JH, rinv, ob = args
    return binv + JH.T @ rinv @ JH

def analysis(xf, binv, y, rinv, htype, gtol=1e-6, method="LBFGS", cgtype=None,
    maxiter=None, restart=False, maxrest=20, 
    disp=False, save_hist=False, save_dh=False, 
    model="model", icycle=0):
    global zetak, alphak
    zetak = []
    alphak = []
    op = htype["operator"]
    pt = htype["perturbation"]
    ga = htype["gamma"]

    JH = obs.dhdx(xf, op, ga)
    ob = y - obs.h_operator(xf, op, ga) 
    nobs = ob.size

    x0 = np.zeros_like(xf)
    args_j = (binv, JH, rinv, ob)
    iprint = np.zeros(2, dtype=np.int32)
    iprint[0] = 1
    minimize = Minimize(x0.size, calc_j, jac=calc_grad_j, hess=calc_hess,
                        args=args_j, iprint=iprint, method=method, cgtype=cgtype,
                        maxiter=maxiter, restart=restart)
    cg = spo.check_grad(calc_j, calc_grad_j, x0, *args_j)
    logger.info("check_grad={}".format(cg))
    if save_hist:
        x, flg = minimize(x0, callback=callback)
        jh = np.zeros(len(zetak))
        gh = np.zeros(len(zetak))
        for i in range(len(zetak)):
            jh[i] = calc_j(np.array(zetak[i]), *args_j)
            g = calc_grad_j(np.array(zetak[i]), *args_j)
            gh[i] = np.sqrt(g.transpose() @ g)
        np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), jh)
        np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), gh)
    else:
        x, flg = minimize(x0)
    
    xa = xf + x
    fun = calc_j(x, *args_j)
    chi2 = fun / nobs

    return xa, chi2