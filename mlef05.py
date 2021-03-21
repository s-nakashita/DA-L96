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
    #if htype["perturbation"] == "grad":
        dh = obs.dhdx(x, htype["operator"]) @ pf
    else:
        dh = obs.h_operator(x[:, None] + pf, htype["operator"]) - hx[:, None]
    return heinv @ zeta - tmat @ dh.transpose() @ rinv @ ob

def calc_hess(zeta, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
    return heinv
    #return np.eye(zeta.size)

def calc_step(alpha_t, zeta, dk, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
    zeta_t = zeta + alpha_t * dk
    x = xc + gmat @ zeta
    x_t = xc + gmat @ zeta_t
    hx = obs.h_operator(x, htype["operator"])
    ob = y - hx
    if htype["perturbation"] == "grad05":
    #if htype["perturbation"] == "grad":
        dh = obs.dhdx(x, htype["operator"]) @ pf
    else:
        dh = obs.h_operator(x[:, None] + pf, htype["operator"]) - hx[:, None]
    cost_t = (zeta_t - zeta).transpose() @ tmat @ dh.transpose() @ rinv @ ob \
             - (zeta_t - zeta).transpose() @ heinv @ zeta
    cost_b = (zeta_t - zeta).transpose() @ heinv @ (zeta_t - zeta) \
             + (zeta_t - zeta).transpose() @ tmat @ dh.transpose() @ rinv @ dh @ tmat @ (zeta_t - zeta)
    return alpha_t * cost_t / cost_b

def minimize_cg(fun, zeta0, args=(), jac=None, calc_step=None, callback=None,
                gtol=1e-5, norm=np.inf, eps=np.finfo(1.0).eps, maxiter=None,
                disp=False):
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

    zeta0 = np.asarray(zeta0).flatten()
    if maxiter is None:
        maxiter = len(zeta0) * 200
    
    old_fval = fun(zeta0, *args)
    gfk = jac(zeta0, *args)
    nfev = 1
    ngev = 1

    k = 0
    zetak = zeta0
    old_old_fval = old_fval + la.norm(gfk) * 0.5

    warnflag = 0
    pk = -gfk
    gnorm = la.norm(gfk, ord=norm)

    sigma_3 = 0.01
    alpha0 = 1.0
    alpha = alpha0
    #if callback is not None:
    #    callback(zetak,alpha)

    while (gnorm > gtol) and (k < maxiter):
        deltak = np.dot(gfk, gfk)
        alpha = calc_step(alpha0, zetak, pk, *args)
        #if alpha > 1.0:
        #    logger.info("max 1.0")
        #    alpha = min(1.0, alpha)
        #if alpha < 0.0:
        #    logger.info("alpha > 0")
        #    alpha = alpha0
        nfev += 1
        if alpha == np.inf:
            warnflag = 2
            break
        zetakp1 = zetak + alpha * pk
        gfkp1 = jac(zetakp1, *args)
        ngev += 1
        yk = gfkp1 - gfk
        beta_k = max(0, np.dot(yk, gfkp1) / deltak)
        pkp1 = -gfkp1 + beta_k * pk
        gnorm = la.norm(gfkp1, ord=norm)

        alpha0 = alpha
        zetak = zetakp1
        pk = pkp1
        gfk = gfkp1
        if callback is not None:
            callback(zetak,alpha)
        logger.debug(f"current:{k} gnorm={gnorm} step-length={alpha}")
        k += 1
    
    fval = fun(zetak, *args)
    nfev += 1
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(zetak).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % nfev)
        print("         Gradient evaluations: %d" % ngev)

    res = OptimizeResult(fun=fval, jac=gfk, nfev=nfev,
                            njev=ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=zetak,
                            nit=k)
    logger.info("success={} message={}".format(res.success, res.message))
    logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
                res.fun, la.norm(res.jac, ord=norm), res.nit))
    return res.x, warnflag

def minimize_bfgs(fun, zeta0, args=(), jac=None, calc_step=None, callback=None,
                gtol=1e-5, norm=np.inf, eps=np.finfo(1.0).eps, maxiter=None,
                disp=False):
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

    zeta0 = np.asarray(zeta0).flatten()
    if maxiter is None:
        maxiter = len(zeta0) * 200
    
    old_fval = fun(zeta0, *args)
    gfk = jac(zeta0, *args)
    nfev = 1
    ngev = 1

    k = 0
    zetak = zeta0
    N = len(zeta0)
    I = np.eye(N, dtype=int)
    Hk = I

    old_old_fval = old_fval + la.norm(gfk) * 0.5

    warnflag = 0
    gnorm = la.norm(gfk, ord=norm)

    sigma_3 = 0.01
    alpha0 = 1.0
    alpha = alpha0
    #if callback is not None:
    #    callback(zetak,alpha)

    while (gnorm > gtol) and (k < maxiter):
        pk = -np.dot(Hk, gfk)
        deltak = np.dot(gfk, gfk)
        alpha = calc_step(alpha0, zetak, pk, *args)
        #if alpha > 1.0:
        #    logger.info("max 1.0")
        #    alpha = min(1.0, alpha)
        #if alpha < 0.0:
        #    logger.info("alpha > 0")
        #    alpha = alpha0
        nfev += 1
        if alpha == np.inf:
            warnflag = 2
            break
        zetakp1 = zetak + alpha * pk
        sk = zetakp1 - zetak
        gfkp1 = jac(zetakp1, *args)
        ngev += 1

        yk = gfkp1 - gfk
        
        alpha0 = alpha
        zetak = zetakp1
        gfk = gfkp1
        gnorm = la.norm(gfkp1, ord=norm)
        old_fval = fun(zetak, *args)
        nfev += 1
        if callback is not None:
            callback(zetak,alpha)
        logger.debug(f"current:{k} gnorm={gnorm} step-length={alpha}")
        k += 1

        if not np.isfinite(old_fval):
            warnflag = 2
            break

        rhok_inv = np.dot(yk, sk)
        if rhok_inv == 0.:
            rhok = 1000.0
            logger.warning("Divide-by-zero encountered : rhok assumed large")
        else:
            rhok = 1. / rhok_inv
        
        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] * sk[np.newaxis, :])
    
    fval = old_fval
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(zetak).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % nfev)
        print("         Gradient evaluations: %d" % ngev)

    res = OptimizeResult(fun=fval, jac=gfk, nfev=nfev,
                            njev=ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=zetak,
                            nit=k)
    logger.info("success={} message={}".format(res.success, res.message))
    logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
                res.fun, la.norm(res.jac, ord=norm), res.nit))
    return res.x, warnflag

def analysis(xf, xc, y, rmat, rinv, htype, gtol=1e-6, method="CG", cgtype=None,
        maxiter=None, maxrest=20, disp=False, save_hist=False, save_dh=False, 
        infl=False, loc = False, infl_parm=1.0, model="z08", icycle=0):
    global zetak, alphak
    zetak = []
    alphak = []
    op = htype["operator"]
    pt = htype["perturbation"]
    pf = xf - xc[:, None]
#    logger.debug("norm(pf)={}".format(la.norm(pf)))
    if infl:
        logger.info("==inflation==, alpha={}".format(infl_parm))
        pf *= infl_parm
    #if pt == "grad":
    if pt == "grad05":
#        logger.debug("dhdx.shape={}".format(obs.dhdx(xc, op).shape))
        dh = obs.dhdx(xc, op) @ pf
    else:
        dh = obs.h_operator(xf, op) - obs.h_operator(xc, op)[:, None]
    if save_dh:
        np.save("{}_dxf_{}_{}_cycle{}.npy".format(model, op, pt, icycle), pf)
        np.save("{}_dh_{}_{}_cycle{}.npy".format(model, op, pt, icycle), dh)
        ob = y - obs.h_operator(xc, op)
        np.save("{}_d_{}_{}_cycle{}.npy".format(model, op, pt, icycle), ob)
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
    if save_dh:
        np.save("{}_tmat_{}_{}_cycle{}.npy".format(model, op, pt, icycle), tmat)
        np.save("{}_pf_{}_{}_cycle{}.npy".format(model, op, pt, icycle), pf)
        np.save("{}_gmat_{}_{}_cycle{}.npy".format(model, op, pt, icycle), gmat)
    x0 = np.zeros(xf.shape[1])
    htype_c = htype.copy()
    Method = method
    #if icycle == 0:
    #    htype_c["perturbation"] = "grad05"
    #    Method="dogleg"
    logger.debug(f"original {htype}")
    logger.debug(f"copy {htype_c}")
        
    args_j = (xc, pf, y, tmat, gmat, heinv, rinv, htype_c)
    iprint = np.zeros(2, dtype=np.int32)
    restart = 0 # restart counter
    flg = -1    # optimization result flag
    #if icycle != 0:
    #    logger.info("calculate gradient")
    minimize = Minimize(x0.size, calc_j, jac=calc_grad_j, hess=calc_hess,
                        args=args_j, iprint=iprint, method=Method, cgtype=cgtype,
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

    if save_hist:
        while restart < maxrest:
            x, flg = minimize(x0, callback=callback)
            if flg == 0:
                logger.info(f"converged at {restart}th restart")
                break 
            restart += 1
            xa = xc + gmat @ x
            xc = xa
            if pt == "grad05":
                dh = obs.dhdx(xc, op) @ pf
            else:
                dh = obs.h_operator(xf, op) - obs.h_operator(xc, op)[:, None]
            zmat = rmat @ dh
            tmat, heinv = precondition(zmat)
            pa = pf @ tmat 
            pf = pa
            gmat = pf @ tmat
            args_j = (xc, pf, y, tmat, gmat, heinv, rinv, htype_c)
            x0 = np.zeros(xf.shape[1])
            #reload(minimize)
            minimize = Minimize(x0.size, calc_j, jac=calc_grad_j, hess=calc_hess,
                        args=args_j, iprint=iprint, method=Method, cgtype=cgtype,
                        maxiter=maxiter, restart=restart)
        #x, flg = minimize_cg(calc_j, x0, args=args_j, jac=calc_grad_j, 
        #                calc_step=calc_step, callback=callback)
        #x, flg = minimize_bfgs(calc_j, x0, args=args_j, jac=calc_grad_j, 
        #                calc_step=calc_step, callback=callback)

        logger.debug(f"zetak={len(zetak)} alpha={len(alphak)}")
        jh = np.zeros(len(zetak))
        gh = np.zeros(len(zetak))
        for i in range(len(zetak)):
            jh[i] = calc_j(np.array(zetak[i]), *args_j)
            g = calc_grad_j(np.array(zetak[i]), *args_j)
            gh[i] = np.sqrt(g.transpose() @ g)
        np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), jh)
        np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), gh)
        np.savetxt("{}_alpha_{}_{}_cycle{}.txt".format(model, op, pt, icycle), alphak)
        if model=="z08":
            if len(zetak) > 0:
                xmax = max(np.abs(np.min(zetak)),np.max(zetak))
            else:
                xmax = max(np.abs(np.min(x)),np.max(x))
            logger.debug("resx max={}".format(xmax))
#            print("resx max={}".format(xmax))
            #if xmax < 1000:
                #cost_j(1000, xf.shape[1], model, x, icycle, *args_j)
                #cost_j2d(1000, xf.shape[1], model, x, icycle, *args_j)
            #    costJ.cost_j2d(1000, xf.shape[1], model, x, icycle, 
            #                   htype, calc_j, calc_grad_j, *args_j)
            #else:
                #xmax = int(xmax*0.01+1)*100
            xmax = np.ceil(xmax*0.01)*100
            logger.debug("resx max={}".format(xmax))
#                print("resx max={}".format(xmax))
                #cost_j(xmax, xf.shape[1], model, x, icycle, *args_j)
                #cost_j2d(xmax, xf.shape[1], model, x, icycle, *args_j)
            costJ.cost_j2d(xmax, xf.shape[1], model, np.array(zetak), icycle,
                               htype, calc_j, calc_grad_j, *args_j)
        elif model=="l96":
            cost_j(200, xf.shape[1], model, x, icycle, *args_j)
    else:
        while restart < maxrest:
            x, flg = minimize(x0)
            if flg == 0:
                logger.info(f"converged at {restart}th restart")
                break 
            restart += 1
            xa = xc + gmat @ x
            xc = xa
            if pt == "grad05":
                dh = obs.dhdx(xc, op) @ pf
            else:
                dh = obs.h_operator(xf, op) - obs.h_operator(xc, op)[:, None]
            zmat = rmat @ dh
            tmat, heinv = precondition(zmat)
            pa = pf @ tmat 
            pf = pa
            gmat = pf @ tmat
            args_j = (xc, pf, y, tmat, gmat, heinv, rinv, htype_c)
            x0 = np.zeros(xf.shape[1])
            #reload(Minimize)
            minimize = Minimize(x0.size, calc_j, jac=calc_grad_j, hess=calc_hess,
                        args=args_j, iprint=iprint, method=Method, cgtype=cgtype,
                        maxiter=maxiter, restart=restart)
        #x = minimize_cg(calc_j, x0, args=args_j, jac=calc_grad_j, 
        #                calc_step=calc_step)
        #x = minimize_bfgs(calc_j, x0, args=args_j, jac=calc_grad_j, 
        #                calc_step=calc_step)
        
    xa = xc + gmat @ x
    if pt == "grad05":
    #if pt == "grad":
        dh = obs.dhdx(xa, op) @ pf
    else:
        dh = obs.h_operator(xa[:, None] + pf, op) - obs.h_operator(xa, op)[:, None]
    zmat = rmat @ dh
#    logger.debug("cond(zmat)={}".format(la.cond(zmat)))
    tmat, heinv = precondition(zmat)
    d = y - obs.h_operator(xa, op)
    chi2 = chi2_test(zmat, heinv, rmat, d)
    pa = pf @ tmat 
    if save_dh:
        logger.info("save ua")
        ua = np.zeros((xa.size,pf.shape[1]+1))
        ua[:,0] = xa
        ua[:,1:] = xa[:, None] + pa
        np.save("{}_ua_{}_{}_cycle{}.npy".format(model, op, pt, icycle), ua)
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