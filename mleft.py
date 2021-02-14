import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
import obs
import costJ


#logging.config.fileConfig("logging_config.ini")
#logger = logging.getLogger(__name__)

def calc_hess(dy, rinv):
    #u, s, vt = la.svd(zmat)
    #v = vt.transpose()
    #is2r = 1 / (1 + s**2)
    #tmat = v @ np.diag(np.sqrt(is2r)) @ vt
    #heinv = v @ np.diag(is2r) @ vt
    c = dy.T @ rinv @ dy
    lam, v = la.eigh(c)
    hes = v @ np.diag(lam + np.ones(lam.size)) @ v.T
    D = np.diag(1.0/(np.sqrt(lam + np.ones(lam.size))))
    tmat = v @ D @ v.T
    Dinv = np.diag((np.sqrt(lam + np.ones(lam.size))))
    tinv = v @ Dinv @ v.T
#    logger.debug("tmat={}".format(tmat))
#    logger.debug("heinv={}".format(heinv))
#    logger.debug("s={}".format(s))
    #print("eigenvalues={}".format(lam))
    #print("cond(hessian)={}".format(la.cond(hes)))
    return tmat, tinv


def callback(xk):
    global zetak
#    logger.debug("xk={}".format(xk))
    zetak.append(xk)


def calc_j(w, *args):
    xf_, dxf, y, calc_hess, rinv, htype = args
    #xf_, dxf, y, tmat, gmat, heinv, rinv, htype = args
    nmem = w.size
    x = xf_ + dxf @ w
    tmat = np.load("tmat.npy")
    tinv = np.load("tinv.npy")
    emat = x[:, None] + np.sqrt(nmem-1)*dxf @ tmat
    hemat = obs.h_operator(emat, htype["operator"], htype["gamma"])
    dy = (hemat - np.mean(hemat, axis=1)[:, None]) @ tinv / np.sqrt(nmem-1)
    ob = y - np.mean(hemat, axis=1)
    j = 0.5 * (w.transpose() @ w + ob.transpose() @ rinv @ ob)
    #j = 0.5 * ((nmem-1)*zeta.transpose() @ heinv @ zeta + nob.transpose() @ rinv @ ob)
#    logger.debug("zeta.shape={}".format(zeta.shape))
#    logger.debug("j={} zeta={}".format(j, zeta))
    return j
    

def calc_grad_j(w, *args):
    xf_, dxf, y, calc_hess, rinv, htype = args
    #xf_, dxf, y, tmat, gmat, heinv, rinv, htype = args
    nmem = w.size
    x = xf_ + dxf @ w
    tmat = np.load("tmat.npy")
    tinv = np.load("tinv.npy")
    emat = x[:, None] + np.sqrt(nmem-1)*dxf @ tmat
    hemat = obs.h_operator(emat, htype["operator"], htype["gamma"])
    dy = (hemat - np.mean(hemat, axis=1)[:, None]) @ tinv / np.sqrt(nmem-1)
    # update tmat & tinv
    tmat, tinv = calc_hess(dy, rinv)
    np.save("tmat.npy", tmat)
    np.save("tinv.npy", tinv)
    ob = y - np.mean(hemat, axis=1)
    return w - dy.transpose() @ rinv @ ob
    #return (nmem-1)*tmat @ zeta - dh.transpose() @ rinv @ ob


def analysis(xf, xf_, y, rmat, rinv, htype, gtol=1e-6,  
        maxiter=None, disp=False, save_hist=False, save_dh=False,
        infl=False, loc = False, tlm=False, infl_parm=1.0, model="z08", icycle=100):
    global zetak
    zetak = []
    op = htype["operator"]
    pt = htype["perturbation"]
    ga = htype["gamma"]
    nmem = xf.shape[1]
    dxf = (xf - xf_[:, None])/np.sqrt(nmem-1)
    condh = np.zeros(2)
    tmat = np.eye(nmem) # Transform matrix
    tinv = np.eye(nmem)
    np.save("tmat.npy", tmat)
    np.save("tinv.npy", tinv)
    if infl:
        """
        if op == "linear" or op == "test":
            if pt == "mlef":
                alpha = 1.2
            else:
                alpha = 1.2
        elif op == "quadratic" or op == "quadratic-nodiff":
            if pt == "mlef":
                alpha = 1.3
            else:
                alpha = 1.35
        elif op == "cubic" or op == "cubic-nodiff":
            if pt == "mlef":
                alpha = 1.6
            else:
                alpha = 1.65
        """
        alpha = infl_parm
        print("==inflation==, alpha={}".format(alpha))
        dxf *= alpha
#    logger.debug("norm(pf)={}".format(la.norm(pf)))
    if loc:
        l_sig = 4.0
        print("==localization==, l_sig={}".format(l_sig))
        dxf = pfloc(dxf, l_sig, save_dh, model, op, pt, icycle)
    emat = xf_[:, None] + np.sqrt(nmem-1) * dxf @ tmat
    hemat = obs.h_operator(emat, op, ga)
    dy = (hemat - np.mean(hemat, axis=1)[:, None]) @ la.inv(tmat) / np.sqrt(nmem-1)
    if save_dh:
        #np.save("{}_dh_{}_{}_cycle{}.npy".format(model, op, pt, icycle), dh)
        np.save("{}_dy_{}_{}_cycle{}.npy".format(model, op, pt, icycle), dy)
        ob = y - np.mean(obs.h_operator(xf, op, ga), axis=1)
        np.save("{}_d_{}_{}_cycle{}.npy".format(model, op, pt, icycle), ob)
#    logger.info("save_dh={}".format(save_dh))
    print("save_dh={}".format(save_dh))
    tmat, tinv = calc_hess(dy, rinv)
    hess = tinv @ tinv.T
    condh[0] = la.cond(hess)
    if icycle == 0:
        print("tmat={}".format(tmat))
        print("tinv={}".format(tinv))
#    logger.debug("pf.shape={}".format(pf.shape))
#    logger.debug("tmat.shape={}".format(tmat.shape))
#    logger.debug("heinv.shape={}".format(heinv.shape))
    #gmat = dxf @ tmat
    #gmat = np.sqrt(nmem-1) * pf @ tmat
#    logger.debug("gmat.shape={}".format(gmat.shape))
    w0 = np.zeros(xf.shape[1])
    args_j = (xf_, dxf, y, calc_hess, rinv, htype)
#    logger.info("save_hist={}".format(save_hist))
    print("save_hist={} cycle{}".format(save_hist, icycle))
    cg = spo.check_grad(calc_j, calc_grad_j, w0, *args_j)
    print("check_grad={}".format(cg))
    if save_hist:
        #g = calc_grad_j(w0, *args_j)
        #print("g={}".format(g))
        res = spo.minimize(calc_j, w0, args=args_j, method='BFGS', \
                jac=calc_grad_j, options={'gtol':gtol, 'maxiter':maxiter, 'disp':disp}, callback=callback)
        #res = spo.minimize(calc_j, x0, args=args_j, method='BFGS', \
        #        jac=None, options={'gtol':gtol, 'disp':disp}, callback=callback)
        jh = np.zeros(len(zetak))
        gh = np.zeros(len(zetak))
        for i in range(len(zetak)):
            jh[i] = calc_j(np.array(zetak[i]), *args_j)
            g = calc_grad_j(np.array(zetak[i]), *args_j)
            gh[i] = np.sqrt(g.transpose() @ g)
        np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), jh)
        np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), gh)
        if model=="z08":
            xmax = max(np.abs(np.min(res.x)),np.max(res.x))
            print("resx max={}".format(xmax))
            if xmax < 1000:
                cost_j(1000, xf.shape[1], model, res.x, icycle, *args_j)
            else:
                xmax = int(xmax*0.01+1)*100
                print("resx max={}".format(xmax))
                cost_j(xmax, xf.shape[1], model, res.x, icycle, *args_j)
        elif model=="l96":
            cost_j(200, xf.shape[1], model, res.x, icycle, *args_j)
    else:
        res = spo.minimize(calc_j, w0, args=args_j, method='BFGS', \
                jac=calc_grad_j, options={'gtol':gtol, 'maxiter':maxiter, 'disp':disp})
        #res = spo.minimize(calc_j, x0, args=args_j, method='BFGS', \
        #        jac=None, options={'gtol':gtol, 'disp':disp})
#    logger.info("success={} message={}".format(res.success, res.message))
    print("success={} message={}".format(res.success, res.message))
#    logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
#            res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
    print("J={:7.3e} dJ={:7.3e} nit={}".format( \
            res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
    print("x={}".format(dxf @ res.x))
    xa_ = xf_ + dxf @ res.x
    condh[1] = la.cond(la.inv(res.hess_inv))
    if save_dh:
        np.save("{}_dx_{}_{}_cycle{}.npy".format(model, op, pt, icycle), dxf@res.x)
    tmat = np.load("tmat.npy")
    tinv = np.load("tinv.npy")
    emat = xa_[:, None] + np.sqrt(nmem-1) * dxf @ tmat
    hemat = obs.h_operator(emat, op, ga)
    dy = (hemat - np.mean(hemat, axis=1)[:, None]) @ tinv / np.sqrt(nmem-1)
    dxa = dxf @ tmat 
    xa = xa_[:, None] + dxa * np.sqrt(nmem-1)
    pa = dxa @ dxa.T
    if save_dh:
        np.save("{}_pa_{}_{}_cycle{}.npy".format(model, op, pt, icycle), pa)
        ua = np.zeros((xa_.size, nmem+1))
        ua[:, 0] = xa_
        ua[:, 1:] = xa
        np.save("{}_ua_{}_{}_cycle{}.npy".format(model, op, pt, icycle), ua)
        #print("xa={}".format(xa))
    d = y - np.mean(obs.h_operator(xa, op, ga), axis=1)
    zmat = rmat @ dy
    heinv = tmat @ tmat.T
    chi2 = chi2_test(zmat, heinv, rmat, d)
    ds = dof(zmat)
    print(ds)
    #if infl:
    #    if op == "linear":
    #        if pt == "mlef":
    #            alpha = 1.1
    #        else:
    #            alpha = 1.2
    #    elif op == "quadratic" or op == "quadratic-nodiff":
    #        if pt == "mlef":
    #            alpha = 1.3
    #        else:
    #            alpha = 1.35
    #    elif op == "cubic" or op == "cubic-nodiff":
    #        if pt == "mlef":
    #            alpha = 1.6
    #        else:
    #            alpha = 1.65
    #    print("==inflation==, alpha={}".format(alpha))
    #    pa *= alpha
        
    return xa, xa_, pa, chi2, ds, cg

def cost_j(nx, nmem, model, xopt, icycle, *args):
    xf_, dxf, y, calc_hess, rinv, htype = args
    #xc, pf, y, tmat, gmat, heinv, rinv, htype = args
    op = htype["operator"]
    pt = htype["perturbation"]
    delta = np.linspace(-nx,nx,4*nx)
    jvalb = np.zeros((len(delta)+1,nmem))
    jvalb[0,:] = xopt
    #jvalo = np.zeros_like(jvalb)
    #jvalo[0,:] = xopt
    #jvalb[1:,:], jvalo[1:,:] = costJ.cost_j(nx, nmem, *args)
    #np.save("{}_cJb_{}_{}_cycle{}.npy".format(model, op, pt, icycle), jvalb)
    #np.save("{}_cJo_{}_{}_cycle{}.npy".format(model, op, pt, icycle), jvalo)
    for k in range(nmem):
        x0 = np.zeros(nmem)
        for i in range(len(delta)):
            x0[k] = delta[i]
            j = calc_j(x0, *args)
            jvalb[i+1,k] = j
    np.save("{}_cJ_{}_{}_cycle{}.npy".format(model, op, pt, icycle), jvalb)

def chi2_test(zmat, heinv, rmat, d):
    p = d.size
    G_inv = np.eye(p) - zmat @ heinv @ zmat.T
    innv = rmat @ d[:,None]
    return innv.T @ G_inv @ innv / p

def dof(zmat):
    u, s, vt = la.svd(zmat)
    ds = np.sum(s**2/(1.0+s**2))
    return ds

def pfloc(sqrtpf, l_sig, save_dh=False, 
model="z08", op="linear", pt="mlef", icycle=0):
    nmem = sqrtpf.shape[1]
    pf = sqrtpf @ sqrtpf.T
    if save_dh:
        np.save("{}_pf_{}_{}_cycle{}.npy".format(model, op, pt, icycle), pf)
        np.save("{}_spf_{}_{}_cycle{}.npy".format(model, op, pt, icycle), sqrtpf)
    dist, l_mat = loc_mat(l_sig, pf.shape[0], pf.shape[1])
    pf = pf * l_mat
    lam, v = la.eig(pf)
    lam[nmem:] = 0.0
    print("eigen value = {}".format(lam))
    pf = v @ np.diag(lam) @ v.T
    spf = v[:,:nmem] @ np.diag(np.sqrt(lam[:nmem]))
    if save_dh:
        np.save("{}_lpf_{}_{}_cycle{}.npy".format(model, op, pt, icycle), pf)
        np.save("{}_lspf_{}_{}_cycle{}.npy".format(model, op, pt, icycle), spf)
    return spf

def loc_mat(sigma, nx, ny):
    dist = np.zeros((nx,ny))
    l_mat = np.zeros_like(dist)
    for j in range(nx):
        for i in range(ny):
            dist[j,i] = min(abs(j-i),nx-abs(j-i))
    d0 = 2.0 * np.sqrt(10.0/3.0) * sigma
    l_mat = np.exp(-dist**2/(2.0*sigma**2))
    l_mat[dist>d0] = 0
    return dist, l_mat 