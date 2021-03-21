import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
import obs
import costJ
from minimize import Minimize


logging.config.fileConfig("./logging_config.ini")
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
    hes = v @ np.diag(lam + np.ones(lam.size)) @ v.T
#    logger.debug("tmat={}".format(tmat))
#    logger.debug("heinv={}".format(heinv))
#    logger.debug("s={}".format(s))
    #print("eigenvalues={}".format(lam))
    #print("cond(hessian)={}".format(la.cond(hes)))
    return tmat, heinv, la.cond(hes)


def callback(xk):
    global zetak
#    logger.debug("xk={}".format(xk))
    zetak.append(xk)


def calc_j(zeta, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
    nmem = zeta.size
    x = xc + gmat @ zeta
    ob = y - obs.h_operator(x, htype["operator"], htype["gamma"])
    j = 0.5 * (zeta.transpose() @ heinv @ zeta + ob.transpose() @ rinv @ ob)
    #j = 0.5 * ((nmem-1)*zeta.transpose() @ heinv @ zeta + nob.transpose() @ rinv @ ob)
#    logger.debug("zeta.shape={}".format(zeta.shape))
#    logger.debug("j={} zeta={}".format(j, zeta))
    return j
    

def calc_grad_j(zeta, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
    nmem = zeta.size
    x = xc + gmat @ zeta
    hx = obs.h_operator(x, htype["operator"], htype["gamma"])
    ob = y - hx
    if htype["perturbation"] == "grad":
        dh = obs.dhdx(x, htype["operator"], htype["gamma"]) @ pf
    else:
        dh = obs.h_operator(x[:, None] + pf, htype["operator"], htype["gamma"]) - hx[:, None]
    return tmat @ zeta - dh.transpose() @ rinv @ ob
    #return (nmem-1)*tmat @ zeta - dh.transpose() @ rinv @ ob


def analysis(xf, xc, y, rmat, rinv, htype, gtol=1e-6, method="LBFGS", cgtype=None,
        maxiter=None, disp=False, save_hist=False, save_dh=False,
        infl=False, loc = False, infl_parm=1.0, model="z08", icycle=100):
    global zetak
    zetak = []
    op = htype["operator"]
    pt = htype["perturbation"]
    ga = htype["gamma"]
    nmem = xf.shape[1]
    pf = xf - xc[:, None]
#    pf = (xf - xc[:, None]) / np.sqrt(nmem)
    #condh = np.zeros(2)
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
        logger.info("==inflation==, alpha={}".format(alpha))
        #print("==inflation==, alpha={}".format(alpha))
        pf *= alpha
#    logger.debug("norm(pf)={}".format(la.norm(pf)))
    if loc:
        l_sig = 4.0
        logger.info("==localization==, l_sig={}".format(l_sig))
        #print("==localization==, l_sig={}".format(l_sig))
        pf = pfloc(pf, l_sig, save_dh, model, op, pt, icycle)
    if pt == "grad":
        logger.debug("dhdx={}".format(obs.dhdx(xc, op, ga)))
        dh = obs.dhdx(xc, op, ga) @ pf
    else:
        dh = obs.h_operator(xf, op, ga) - obs.h_operator(xc, op, ga)[:, None]
    if save_dh:
        #np.save("{}_dh_{}_{}_cycle{}.npy".format(model, op, pt, icycle), dh)
        np.save("{}_dy_{}_{}_cycle{}.npy".format(model, op, pt, icycle), dh)
        ob = y - obs.h_operator(xc, op, ga)
        np.save("{}_d_{}_{}_cycle{}.npy".format(model, op, pt, icycle), ob)
    logger.info("save_dh={}".format(save_dh))
#    print("save_dh={}".format(save_dh))
    zmat = rmat @ dh
#    logger.debug("cond(zmat)={}".format(la.cond(zmat)))
    tmat, heinv, condh = precondition(zmat)
    #if icycle == 0:
    #    print("tmat={}".format(tmat))
    #    print("heinv={}".format(heinv))
#    logger.debug("pf.shape={}".format(pf.shape))
#    logger.debug("tmat.shape={}".format(tmat.shape))
#    logger.debug("heinv.shape={}".format(heinv.shape))
    gmat = pf @ tmat
    #gmat = np.sqrt(nmem-1) * pf @ tmat
#    logger.debug("gmat.shape={}".format(gmat.shape))
    x0 = np.zeros(xf.shape[1])
    args_j = (xc, pf, y, tmat, gmat, heinv, rinv, htype)
    iprint = np.zeros(2, dtype=np.int32)
    iprint[0] = 1
    minimize = Minimize(x0.size, calc_j, jac=calc_grad_j, 
                        args=args_j, iprint=iprint, method=method, cgtype=cgtype)
    logger.info("save_hist={}".format(save_hist))
#    print("save_hist={} cycle{}".format(save_hist, icycle))
    cg = spo.check_grad(calc_j, calc_grad_j, x0, *args_j)
    logger.info("check_grad={}".format(cg))
    if save_hist:
        x = minimize(x0,callback=callback)
        logger.debug(zetak)
#        print(len(zetak))
        jh = np.zeros(len(zetak))
        gh = np.zeros(len(zetak))
        for i in range(len(zetak)):
            jh[i] = calc_j(np.array(zetak[i]), *args_j)
            g = calc_grad_j(np.array(zetak[i]), *args_j)
            gh[i] = np.sqrt(g.transpose() @ g)
        np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), jh)
        np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), gh)
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
            costJ.cost_j2d(xmax, xf.shape[1], model, x, icycle,
                               htype, calc_j, calc_grad_j, *args_j)
        elif model=="l96":
            cost_j(200, xf.shape[1], model, x, icycle, *args_j)
    else:
        x = minimize(x0)
    xa = xc + gmat @ x
    
    if save_dh:
        np.save("{}_dx_{}_{}_cycle{}.npy".format(model, op, pt, icycle), gmat@x)
    if pt == "grad":
        dh = obs.dhdx(xa, op, ga) @ pf
    else:
        dh = obs.h_operator(xa[:, None] + pf, op, ga) - obs.h_operator(xa, op, ga)[:, None]
    zmat = rmat @ dh
#    logger.debug("cond(zmat)={}".format(la.cond(zmat)))
    tmat, heinv, dum = precondition(zmat)
    pa = pf @ tmat 
#    pa *= np.sqrt(nmem)
    if save_dh:
        np.save("{}_pa_{}_{}_cycle{}.npy".format(model, op, pt, icycle), pa)
        ua = np.zeros((xa.size, nmem+1))
        ua[:, 0] = xa
        ua[:, 1:] = xa[:, None] + pa
        np.save("{}_ua_{}_{}_cycle{}.npy".format(model, op, pt, icycle), ua)
        #print("xa={}".format(xa))
    d = y - obs.h_operator(xa, op, ga)
    chi2 = chi2_test(zmat, heinv, rmat, d)
    ds = dof(zmat)
    logger.info("DOF = {}".format(ds))
        
    return xa, pa, chi2, ds, condh

def cost_j(nx, nmem, model, xopt, icycle, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
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

def cost_j2d(xmax, nmem, model, xopt, icycle, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
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
    logger.debug("eigen value = {}".format(lam))
#    print("eigen value = {}".format(lam))
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