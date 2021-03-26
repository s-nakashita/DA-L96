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
    prec = v @ np.diag(lam + np.ones(lam.size)) @ v.T
    tmat = v @ np.diag(1.0/np.sqrt(lam + np.ones(lam.size))) @ v.T
    heinv = tmat @ tmat.T
    logger.debug("heinv={}".format(heinv))
#    logger.debug("s={}".format(s))
    #print("s={}".format(s))
    return tmat, heinv, prec


wk = []
alphak = []
def callback(xk, alpha=None):
    global wk, alphak
#    logger.debug("xk={}".format(xk))
    wk.append(xk)
    if alpha is not None:
        alphak.append(alpha)


def calc_j(w, *args):
    xc, pf, y, rmat, htype = args
    rinv = rmat @ rmat.transpose()
    x = xc + pf @ w
    ob = y - obs.h_operator(x, htype["operator"])
    j = 0.5 * (w.transpose() @ w + ob.transpose() @ rinv @ ob)
#    logger.debug("zeta.shape={}".format(zeta.shape))
#    logger.debug("j={} zeta={}".format(j, zeta))
    return j
    

def calc_grad_j(w, *args):
    xc, pf, y, rmat, htype = args
    rinv = rmat @ rmat.transpose()
    x = xc + pf @ w
    hx = obs.h_operator(x, htype["operator"])
    ob = y - hx
    if htype["perturbation"] == "gradw":
        dh = obs.dhdx(x, htype["operator"]) @ pf
    else:
        dh = obs.h_operator(x[:, None] + pf, htype["operator"]) - hx[:, None]
    return w - dh.transpose() @ rinv @ ob

def calc_hess(w, *args):
    xc, pf, y, rmat, htype = args
    rinv = rmat @ rmat.transpose()
    x = xc + pf @ w
    if htype["perturbation"] == "gradw":
        dh = obs.dhdx(x, htype["operator"]) @ pf
    else:
        dh = obs.h_operator(x[:, None] + pf, htype["operator"]) - obs.h_operator(x, htype["operator"])[:, None]
    return np.eye(w.size) + dh.transpose() @ rinv @ dh

def minimize_newton(fun, w0, args=(), jac=None, hess=None, callback=None, 
                    gtol=1e-5, maxiter=None, disp=False, 
                    delta=1e-10, mu=0.5, c1=1e-3, c2=0.9):
    _status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}
    w0 = np.asarray(w0).flatten()
    if maxiter is None:
        maxiter = len(w0) * 200
    
    old_fval = fun(w0, *args)
    gfk = jac(w0, *args)
    nfev = 1
    ngev = 1
    k = 0
    wk = w0

    warnflag = 0
    gnorm = np.sqrt(np.dot(gfk, gfk))
    phik = min(mu,np.sqrt(gnorm))

    xc, pf, y, rmat, htype = args
    if htype["perturbation"] == "gradw":
        dh = obs.dhdx(xc, htype["operator"]) @ pf
    else:
        dh = obs.h_operator(xc[:, None] + pf, htype["operator"]) - obs.h_operator(xc, htype["operator"])[:, None]
    zmat = rmat @ dh
    tmat, heinv, Mk = precondition(zmat)
    #Mk = np.eye(w0.size)
    logger.debug(f"preconditioner={Mk}")

    while (gnorm > gtol) and (k < maxiter):
        Hk = hess(wk, *args)
        lam, v = la.eigh(Hk)
        #lam[:len(lam)-1] = 0.0
        #Mk = v @ np.diag(lam) @ v.transpose()
        logger.debug(f"Hessian eigen values={lam.max(), lam.min()}")
        #pk = la.solve(Hk, -gfk)
        #eps = phik * gnorm
        eps = 1e-8
        pk = pcg(gfk, Hk, Mk, delta=delta, eps=eps)
        rk = Hk @ pk + gfk
        logger.debug(f"residual:{np.sqrt(np.dot(rk, rk))}")
        #alphak = back_tracking(fun, wk, pk, gfk, old_fval, nfev, args,
        #                       c1, c2)
        alphak = 1.0
        wkp1 = wk + alphak * pk
        gfkp1 = jac(wkp1, *args)
        gfk = gfkp1
        wk = wkp1
        ngev += 1
        gnorm = np.sqrt(np.dot(gfk, gfk))
        old_fval = fun(wkp1, *args)
        nfev += 1
        if callback is not None:
            callback(wk, alphak)
        logger.debug(f"current:{k} gnorm={gnorm} step-length={alphak} phik={eps}")
        k += 1
        phik = min(mu/k, np.sqrt(gnorm))
    fval = fun(wk, *args)
    nfev += 1
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(wk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        logger.info("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        logger.info("         Current function value: %f" % fval)
        logger.info("         Iterations: %d" % k)
        logger.info("         Function evaluations: %d" % nfev)
        logger.info("         Gradient evaluations: %d" % ngev)
    else:
        logger.info("success={} message={}".format(warnflag==0, msg))
        logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
                fval, gnorm, k))
    return wk

def pcg(g, H, M, delta=1e-10, eps=None, maxiter=30):
    j = 0
    pj = np.zeros_like(g)
    rj = g
    rnorm = np.sqrt(np.dot(rj, rj))
    zj = la.solve(M, rj)
    dj = -zj
    p = -g
    while (j < maxiter):
        # negative curvature test
        hd = H @ dj
        d2norm = np.dot(dj, dj)
        if np.dot(dj, hd) <= delta*d2norm:
            logger.debug("satisfy negative curvature")
            break
        # conjugate gradient update
        old_rj = np.dot(rj, zj)
        alpha = old_rj / np.dot(dj, hd)
        pj += alpha * dj
        p = pj
        rj += alpha * hd
        rnorm = np.sqrt(np.dot(rj, rj))
        if rnorm < eps:
            logger.debug("truncate")
            break
        zj = la.solve(M, rj)
        beta = np.dot(rj, zj) / old_rj
        dj = -zj + beta * dj
        j += 1
    logger.debug(f"linear CG terminate at {j}th iteration")
    return p 

def back_tracking(fun, wk, pk, gfk, old_fval, nfev, args, c1, c2):
    alpha0 = 1.0
    while (alpha0 > 1e-3):
        wtrial = wk + alpha0 * pk
        ftrial = fun(wtrial, *args)
        nfev += 1
        if (ftrial - old_fval <= c1*alpha0*np.dot(gfk, pk)) \
            and (ftrial - old_fval >= c2*alpha0*np.dot(gfk, pk)):
            break
        alpha0 *= 0.5
    return alpha0


def analysis(xf, xc, y, rmat, rinv, htype, gtol=1e-6, method="LBFGS",
       maxiter=None, disp=False, save_hist=False, save_dh=False, 
       infl=False, loc = False, infl_parm=1.0, model="z08", icycle=0):
    global wk, alphak
    wk = []
    alphak = []
    op = htype["operator"]
    pt = htype["perturbation"]
    nmem = xf.shape[1]
    pf = xf - xc[:, None]
    #pf = (xf - xc[:, None]) / np.sqrt(nmem)
#    logger.debug("norm(pf)={}".format(la.norm(pf)))
    if infl:
        logger.info("==inflation==, alpha={}".format(infl_parm))
        pf *= infl_parm
    if pt == "gradw":
#        logger.debug("dhdx.shape={}".format(obs.dhdx(xc, op).shape))
        dh = obs.dhdx(xc, op) @ pf
    else:
        dh = obs.h_operator(xf, op) - obs.h_operator(xc, op)[:, None]
    if save_dh:
        np.save("{}_dh_{}_{}.npy".format(model, op, pt), dh)
    logger.info("save_dh={}".format(save_dh))
#    print("save_dh={}".format(save_dh))
    x0 = np.zeros(xf.shape[1])
    args_j = (xc, pf, y, rmat, htype)
    #iprint = np.zeros(2, dtype=np.int32)
    #minimize = Minimize(x0.size, 7, calc_j, calc_grad_j, 
    #                    args_j, iprint, method)
    logger.info("save_hist={}".format(save_hist))
    cg = spo.check_grad(calc_j, calc_grad_j, x0, *args_j)
    logger.info("check_grad={}".format(cg))
#    print("save_hist={}".format(save_hist))
    if save_hist:
        #g = calc_grad_j(x0, *args_j)
        #print("g={}".format(g))
        #x = minimize(x0, callback=callback)
        x = minimize_newton(calc_j, x0, args=args_j, jac=calc_grad_j, hess=calc_hess,
                            callback=callback, maxiter=maxiter, disp=disp)
        logger.debug(f"wk={len(wk)} alpha={len(alphak)}")

        jh = np.zeros(len(wk))
        gh = np.zeros(len(wk))
        for i in range(len(wk)):
            jh[i] = calc_j(np.array(wk[i]), *args_j)
            g = calc_grad_j(np.array(wk[i]), *args_j)
            gh[i] = np.sqrt(g.transpose() @ g)
        np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), jh)
        np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), gh)
        np.savetxt("{}_alpha_{}_{}_cycle{}.txt".format(model, op, pt, icycle), alphak)
        if model=="z08":
            xmax = max(np.abs(np.min(x)),np.max(x))
            logger.debug("resx max={}".format(xmax))
#            print("resx max={}".format(xmax))
            xmax = np.ceil(xmax*0.01)*100
            logger.debug("resx max={}".format(xmax))
#           print("resx max={}".format(xmax))
            #cost_j(xmax, xf.shape[1], model, x, icycle, *args_j)
            #cost_j2d(xmax, xf.shape[1], model, x, icycle, *args_j)
            costJ.cost_j2d(xmax, xf.shape[1], model, x, icycle,
                           htype, calc_j, calc_grad_j, *args_j)
        elif model=="l96":
            cost_j(200, xf.shape[1], model, x, icycle, *args_j)
    else:
        #x = minimize(x0)
        x = minimize_newton(calc_j, x0, args=args_j, jac=calc_grad_j, hess=calc_hess,
                            maxiter=maxiter, disp=disp)
    xa = xc + pf @ x
    
    if pt == "gradw":
        dh = obs.dhdx(xa, op) @ pf
    else:
        dh = obs.h_operator(xa[:, None] + pf, op) - obs.h_operator(xa, op)[:, None]
    zmat = rmat @ dh
#    logger.debug("cond(zmat)={}".format(la.cond(zmat)))
    tmat, heinv, prec = precondition(zmat)
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