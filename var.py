import sys
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
import obs

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

def analysis(xf, binv, y, rinv, htype, gtol=1e-6, maxiter=None, \
    disp=False, save_hist=False, save_dh=False, \
    model="model", icycle=0)
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
    if save_hist:
        res = spo.minimize(calc_j, x0, args=args_j, method='BFGS',\
            jac=calc_grad_j, options={'gtol':gtol, 'maxiter':maxiter, 'disp':disp}, callback=callback)
        jh = np.zeros(len(zetak))
        gh = np.zeros(len(zetak))
        for i in range(len(zetak)):
            jh[i] = calc_j(np.array(zetak[i]), *args_j)
            g = calc_grad_j(np.array(zetak[i]), *args_j)
            gh[i] = np.sqrt(g.transpose() @ g)
        np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), jh)
        np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), gh)
    else:
        res = spo.minimize(calc_j, x0, args=args_j, method='BFGS',\
            jac=calc_grad_j, options={'gtol':gtol, 'maxiter':maxiter, 'disp':disp})
    print("success={} message={}".format(res.success, res.message))
    print("J={:7.3e} dJ={:7.3e} nit={}".format( \
            res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))

    xa = xf + res.x
    chi2 = res.fun / nobs

    return xa, chi2