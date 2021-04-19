import sys
import logging
from logging.config import fileConfig
from importlib import reload
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
from scipy.optimize import OptimizeResult
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
    #logger.debug("tmat={}".format(tmat))
    #logger.debug("heinv={}".format(heinv))
    logger.info("max eigen value ={}".format(np.max(lam)))
#    logger.debug("s={}".format(s))
#    print("s={}".format(s))
    return tmat, heinv


zetak = []
alphak = []
def callback(xk, alpha=None):
    global zetak, alphak
#    logger.debug("xk={}".format(xk))
    zetak.append(xk)
    if alpha is not None:
        alphak.append(alpha)

def calc_j(zeta, *args):
    xc, pf, y, ytype, tmat, gmat, heinv, rinv, htype = args
    x = xc + gmat @ zeta
    ob = np.zeros_like(y)
    for i in range(len(ytype)):
        ob[i] = y[i] - obs.h_operator(x, ytype[i])
    j = 0.5 * (zeta.transpose() @ heinv @ zeta + ob.transpose() @ rinv @ ob)
#    logger.debug("zeta.shape={}".format(zeta.shape))
#    logger.debug("j={} zeta={}".format(j, zeta))
    return j
    

def calc_grad_j(zeta, *args):
    xc, pf, y, ytype, tmat, gmat, heinv, rinv, htype = args
    x = xc + gmat @ zeta
    hx = np.zeros_like(y)
    for i in range(len(ytype)):
        hx[i] = obs.h_operator(x, ytype[i])
    ob = y - hx
    dh = np.zeros((y.size,pf.shape[1]))
    if htype["perturbation"] == "grad05":
    #if htype["perturbation"] == "grad":
        for i in range(len(ytype)):
            dh[i,:] = obs.dhdx(x, ytype[i]) @ pf
    else:
        for i in range(len(ytype)):
            dh[i,:] = obs.h_operator(x[:, None] + pf, ytype[i]) - obs.h_operator(x, ytype[i])[:, None]
    return heinv @ zeta - tmat @ dh.transpose() @ rinv @ ob

def calc_hess(zeta, *args):
    xc, pf, y, ytype, tmat, gmat, heinv, rinv, htype = args
    x = xc + gmat @ zeta
    dh = np.zeros((y.size,pf.shape[1]))
    if htype["perturbation"] == "grad05":
    #if htype["perturbation"] == "grad":
        for i in range(len(ytype)):
            dh[i,:] = obs.dhdx(x, ytype[i]) @ pf
    else:
        for i in range(len(ytype)):
            dh[i,:] = obs.h_operator(x[:, None] + pf, ytype[i]) - obs.h_operator(x, ytype[i])[:, None]
    hess = np.eye(zeta.size) + dh.transpose() @ rinv @ dh
    hess = tmat @ hess @ tmat
    logger.debug(f"hess={hess}")
    return hess
    #return heinv
    #return np.eye(zeta.size)

def analysis(xf, xc, y, ytype, rmat, rinv, htype, gtol=1e-6, method="CG", cgtype=None,
        maxiter=None, restart=True, maxrest=20, 
        disp=False, save_hist=False, save_dh=False, 
        infl=False, loc = False, infl_parm=1.0, model="z08", icycle=0):
    global zetak, alphak
    zetak = []
    alphak = []
    #op = htype["operator"]
    pt = htype["perturbation"]
    pf = xf - xc[:, None]
#    logger.debug("norm(pf)={}".format(la.norm(pf)))
    if infl:
        logger.info("==inflation==, alpha={}".format(infl_parm))
        pf *= infl_parm
    #if pt == "grad":
    dh = np.zeros((y.size,pf.shape[1]))
    if pt == "grad05":
#        logger.debug("dhdx.shape={}".format(obs.dhdx(xc, op).shape))
        for i in range(len(ytype)):
            op = ytype[i]
            dh[i,:] = obs.dhdx(xc, op) @ pf
        #dh = obs.dhdx(xc, op) @ pf
    else:
        for i in range(len(ytype)):
            op = ytype[i]
            dh[i,:] = obs.h_operator(xf, op) - obs.h_operator(xc, op)[:, None]
        #dh = obs.h_operator(xf, op) - obs.h_operator(xc, op)[:, None]
    zmat = rmat @ dh
#    logger.debug("cond(zmat)={}".format(la.cond(zmat)))
    tmat, heinv = precondition(zmat)
#    logger.debug("pf.shape={}".format(pf.shape))
#    logger.debug("tmat.shape={}".format(tmat.shape))
#    logger.debug("heinv.shape={}".format(heinv.shape))
    gmat = pf @ tmat
#    logger.debug("gmat.shape={}".format(gmat.shape))
    x0 = np.zeros(xf.shape[1])
    x = x0
        
    args_j = (xc, pf, y, ytype, tmat, gmat, heinv, rinv, htype)
    iprint = np.zeros(2, dtype=np.int32)
    irest = 0 # restart counter
    flg = -1    # optimization result flag
    #if icycle != 0:
    #    logger.info("calculate gradient")
    minimize = Minimize(x0.size, calc_j, jac=calc_grad_j, hess=calc_hess,
                        args=args_j, iprint=iprint, method=method, cgtype=cgtype,
                        maxiter=maxiter, restart=restart)
    #else:
    #    logger.info("finite difference approximation")
    #    minimize = Minimize(x0.size, calc_j, jac=None, hess=None,
    #                    args=args_j, iprint=iprint, method=Method, cgtype=cgtype,
    #                    maxiter=maxiter)
    logger.info("save_hist={}".format(save_hist))
#    print("save_hist={}".format(save_hist))
    cg = spo.check_grad(calc_j, calc_grad_j, x0, *args_j)
    logger.info("check_grad={}".format(cg))
    if restart:
        if save_hist:
            jh = []
            gh = []         
        while irest < maxrest:
            zetak = []
            xold = x
            if save_hist:
                x, flg = minimize(x0, callback=callback)
            else:
                x, flg = minimize(x0)
            irest += 1
            if save_hist:
                for i in range(len(zetak)):
                    jh.append(calc_j(np.array(zetak[i]), *args_j))
                    g = calc_grad_j(np.array(zetak[i]), *args_j)
                    gh.append(np.sqrt(g.transpose() @ g))
            if flg == 0:
                logger.info(f"Converged at {irest}th restart")
                break 
            xup = x - xold
            #logger.debug(f"update : {xup}")
            if np.sqrt(np.dot(xup,xup)) < 1e-10:
                logger.info(f"Stagnation at {irest}th : solution not updated")
                break
                
            xa = xc + gmat @ x
            xc = xa
            if pt == "grad05":
    #if pt == "grad":
                for i in range(len(ytype)):
                    op = ytype[i]
                    dh[i,:] = obs.dhdx(xc, op) @ pf
        #dh = obs.dhdx(xa, op) @ pf
            else:
                for i in range(len(ytype)):
                    op = ytype[i]
                    dh[i,:] = obs.h_operator(xc[:, None] + pf, op) - obs.h_operator(xc, op)[:, None]
            zmat = rmat @ dh
            tmat, heinv = precondition(zmat)
            #pa = pf @ tmat 
            #pf = pa
            gmat = pf @ tmat
            args_j = (xc, pf, y, ytype, tmat, gmat, heinv, rinv, htype)
            x0 = np.zeros(xf.shape[1])
            #reload(minimize)
            minimize = Minimize(x0.size, calc_j, jac=calc_grad_j, hess=calc_hess,
                    args=args_j, iprint=iprint, method=method, cgtype=cgtype,
                    maxiter=maxiter, restart=restart)
    else:
        #x, flg = minimize_cg(calc_j, x0, args=args_j, jac=calc_grad_j, 
        #                calc_step=calc_step, callback=callback)
        #x, flg = minimize_bfgs(calc_j, x0, args=args_j, jac=calc_grad_j, 
        #                calc_step=calc_step, callback=callback)
        if save_hist:
            x, flg = minimize(x0, callback=callback)
            jh = np.zeros(len(zetak))
            gh = np.zeros(len(zetak))
            for i in range(len(zetak)):
                jh[i] = calc_j(np.array(zetak[i]), *args_j)
                g = calc_grad_j(np.array(zetak[i]), *args_j)
                gh[i] = np.sqrt(g.transpose() @ g)
            logger.debug(f"zetak={len(zetak)} alpha={len(alphak)}")
            np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), jh)
            np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), gh)
            np.savetxt("{}_alpha_{}_{}_cycle{}.txt".format(model, op, pt, icycle), alphak)
        else:
            x, flg = minimize(x0)
        
    xa = xc + gmat @ x
    if pt == "grad05":
    #if pt == "grad":
        for i in range(len(ytype)):
            op = ytype[i]
            dh[i,:] = obs.dhdx(xa, op) @ pf
        #dh = obs.dhdx(xa, op) @ pf
    else:
        for i in range(len(ytype)):
            op = ytype[i]
            dh[i,:] = obs.h_operator(xa[:, None] + pf, op) - obs.h_operator(xa, op)[:, None]
    zmat = rmat @ dh
#    logger.debug("cond(zmat)={}".format(la.cond(zmat)))
    tmat, heinv = precondition(zmat)
    d = np.zeros_like(y)
    for i in range(len(ytype)):
        op = ytype[i]
        d[i] = y[i] - obs.h_operator(xa, op)
    chi2 = chi2_test(zmat, heinv, rmat, d)
    pa = pf @ tmat 
    
    return xa, pa, chi2

def chi2_test(zmat, heinv, rmat, d):
    p = d.size
    G_inv = np.eye(p) - zmat @ heinv @ zmat.T
    innv = rmat @ d[:,None]
    return innv.T @ G_inv @ innv / p