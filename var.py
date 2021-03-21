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

def callback(xk):
    global zetak
    zetak.append(xk)

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

def analysis(xf, binv, y, rinv, htype, gtol=1e-6, method="LBFGS", maxiter=None, 
    disp=False, save_hist=False, save_dh=False, 
    model="model", icycle=0):
    global zetak
    zetak = []
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
    minimize = Minimize(x0.size, 7, calc_j, calc_grad_j,
                        args_j, iprint, method)
    cg = spo.check_grad(calc_j, calc_grad_j, x0, *args_j)
    logger.info("check_grad={}".format(cg))
    if save_hist:
        x = minimize(x0, callback=callback)
        jh = np.zeros(len(zetak))
        gh = np.zeros(len(zetak))
        for i in range(len(zetak)):
            jh[i] = calc_j(np.array(zetak[i]), *args_j)
            g = calc_grad_j(np.array(zetak[i]), *args_j)
            gh[i] = np.sqrt(g.transpose() @ g)
        np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), jh)
        np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), gh)
    else:
        x = minimize(x0)
    
    xa = xf + x
    fun = calc_j(x, *args_j)
    chi2 = fun / nobs

    return xa, chi2