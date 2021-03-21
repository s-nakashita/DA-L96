from lbfgs import lbfgs
from cg import cgfam
import numpy as np
import scipy.optimize as spo
import logging
from logging.config import fileConfig

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')

# standard status messages of optimizers
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
class Minimize():
    def __init__(self, n, func, jac=None, hess=None, args=None, 
        iprint=np.array([0,0]), method="LBFGS", cgtype=None, maxiter=None,
        restart=None):
        self.n = n
        self.m = 7
        self.func = func
        self.jac = jac
        self.hess = hess
        self.args = args
        self.method = method
        # for lbfgs and cgfam
        self.iprint = iprint
        self.eps = 1.0e-5
        self.xtol = 1.0e-16
        # for lbfgs
        self.diagco = False
        self.diag = np.ones(self.n)
        self.llwork = self.n*(2*self.m+1)+2*self.m
        self.lwork = np.zeros(self.llwork)
        # for cgfam
        self.desc = np.ones(self.n)
        self.irest = 1
        # self.cgtype = 1 : Fletcher-Reeves
        #               2 : Polak-Ribiere
        #               3 : Positive Polak-Ribiere
        self.cgtype = cgtype
        self.lcwork = self.n
        self.cwork = np.zeros(self.lcwork)
        # for scipy.optimize.minimize
        self.gtol = 1.0e-6
        self.disp = False
        self.maxiter = maxiter
        logger.info(f"method={self.method}")
        if self.cgtype is not None:
            logger.info("%s%s" % ("cgtype: ", "Fletcher-Reeves" if self.cgtype == 1 else
                                              "Polak-Ribiere" if self.cgtype == 2 else
                                              "Positive Polak-Ribiere" if self.cgtype == 3
                                              else ""))
        self.restart = restart
        logger.info(f"restart={self.restart}")
         
    def __call__(self, x0, callback=None):
        if self.method == "LBFGS":
            return self.minimize_lbfgs(x0, callback=callback)
        elif self.method == "CGF":
            return self.minimize_cgf(x0, callback=callback)
        elif self.method == "GD":
            return self.minimize_gd(x0, callback=callback)
        else:
            return self.minimize_scipy(x0, callback=callback)

    def minimize_gd(self, x0, callback=None):
        from scipy.optimize import line_search
        from scipy.optimize.linesearch import LineSearchWarning
        
        if self.args is not None:
            old_fval = self.func(x0, *self.args)
            gfk = self.jac(x0, *self.args)
        else:
            old_fval = self.func(x0)
            gfk = self.jac(x0)
        k = 0
        xk = x0
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2

        if self.maxiter is None:
            #maxiter = len(x0) * 1000
            maxiter = 1
        else:
            maxiter = self.maxiter
        warnflag = 0
        pk = -gfk
        gnorm = np.linalg.norm(gfk)
        nfev = 1
        ngev = 1
        while (gnorm > self.gtol) and (k < maxiter):
            if self.args is not None:
                alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                    line_search(self.func, self.jac, xk, pk, gfk=gfk,
                    old_fval=old_fval, old_old_fval=old_old_fval,\
                    args=self.args ,amax=1e20)
            else:
                alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                    line_search(self.func, self.jac, xk, pk, gfk=gfk,
                    old_fval=old_fval, old_old_fval=old_old_fval,\
                    amax=1e20)
            if alpha_k is None:
                warnflag = 2
                break
            alpha_k = 1.0
            logger.debug("alpha_k={}".format(alpha_k))
            nfev += fc
            ngev += gc
            xk = xk + alpha_k * pk
            if self.args is not None:
                gfk = self.jac(xk, *self.args)
            else:
                gfk = self.jac(xk)
            ngev += 1
            pk = -gfk
            gnorm = np.linalg.norm(gfk)
            if callback is not None:
                callback(xk,alpha_k)
            k += 1
        if self.args is not None:
            fval = self.func(xk, *self.args)
        else:
            fval = self.func(xk)
        if warnflag == 2:
            msg = _status_message['pr_loss']
        elif k >= maxiter:
            warnflag = 1
            msg = _status_message['maxiter']
        elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
            warnflag = 3
            msg = _status_message['nan']
        else:
            msg = _status_message['success']

        if self.disp:
            logger.info("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
            logger.info("         Current function value: %f" % fval)
            logger.info("         Iterations: %d" % k)
            logger.info("         Function evaluations: %d" % nfev)
            logger.info("         Gradient evaluations: %d" % ngev)
        else:
            logger.info("success={} message={}".format(warnflag==0, msg))
            logger.info("J={:7.3e} nit={}".format(fval, k))
        return xk, warnflag

    def minimize_lbfgs(self, x0, callback=None):
        icall = 0
        iflag = 0

        fval = 0.0
        gval = np.zeros_like(x0)
        if self.args != None:
            fval = self.func(x0, *self.args)
            gval = self.jac(x0, *self.args)
        else:
            fval = self.func(x0)
            gval = self.jac(x0)
        logger.info("initial function value = {}".format(fval))
#        print("initial function value = {}".format(fval))
        logger.info("initial gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
#        print("initial gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))

        x = x0.copy()
        #alpha = 1.0
        #if callback != None:
        #    callback(x, alpha)
        if self.maxiter is None:
            maxiter = 20
        else:
            maxiter = self.maxiter
        # check stagnation
        nomove = 0
        for icall in range(maxiter):
            [xk, alphak, oflag] = lbfgs(n=self.n, m=self.m, x=x, f=fval, g=gval, \
                          diagco=self.diagco, diag=self.diag, \
                          iprint=self.iprint, eps=self.eps, xtol=self.xtol, w=self.lwork, \
                          iflag=iflag, irest=self.irest)
            iflag = oflag
            #update = np.dot((xk - x),(xk - x))
            #logger.debug(f"update={update}")
            x = xk[:]
            if iflag == 1:
                #if update < 1e-5:
                #    nomove += 1
                #    if nomove > 2:
                #        logger.info("stagnation for more than 3 continuous iterations")
                #        iflag = -1
                #        break
                #else:
                #    nomove = 0 # reset stagnation counter
                if callback != None:
                    callback(x, alphak)
            if self.args != None:
                fval = self.func(x, *self.args)
                gval = self.jac(x, *self.args)
            else:
                fval = self.func(x)
                gval = self.jac(x)
            if iflag == 0:
                if callback != None:
                    callback(x, alphak)
                logger.info("minimization success")
                logger.info("iteration = {}".format(icall))
#                print("iteration = {}".format(icall))
                logger.info("final function value = {}".format(fval))
#                print("final function value = {}".format(fval))
                logger.info("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
#                print("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
                break
            if iflag <= 0:
                if callback != None:
                    callback(x, alphak)
                logger.info("minimization failed, FLAG = {}".format(iflag))
                logger.info("iteration = {}".format(icall))
#                print("iteration = {}".format(icall))
                logger.info("final function value = {}".format(fval))
#                print("final function value = {}".format(fval))
                logger.info("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
#                print("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
                break
        if iflag > 0:
            logger.info("minimization not converged")
#            print("minimization not converged")
            logger.info("current function value = {}".format(fval))
#            print("current function value = {}".format(fval))
            logger.info("current gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
#            print("current gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))

        return x, iflag

    def minimize_cgf(self, x0, callback=None):
        icall = 0
        iflag = 0

        fval = 0.0
        gval = np.zeros_like(x0)
        if self.args != None:
            fval = self.func(x0, *self.args)
            gval = self.jac(x0, *self.args)
        else:
            fval = self.func(x0)
            gval = self.jac(x0)
        logger.info("initial function value = {}".format(fval))
#        print("initial function value = {}".format(fval))
        logger.info("initial gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
#        print("initial gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))

        x = x0.copy()
        xold = x.copy()
        gold = gval.copy()
        gold_old = gold.copy()
        dold = self.desc.copy()
        finish = False
        #if callback != None:
        #    callback(x)
        if self.maxiter is None:
            maxiter = 20
        else:
            maxiter = self.maxiter
        while icall < maxiter:
            [x, gval, self.desc, gold, alphak, oflag, ofinish] = \
                cgfam(n=self.n, x=xold, f=fval, g=gold, \
                    d=dold, gold=gold_old, \
                    iprint=self.iprint, eps=self.eps, w=self.cwork, iflag=iflag, \
                    irest=self.irest, method=self.cgtype, finish=finish)
            iflag = oflag
            finish = bool(ofinish==1)
            xold = x[:]
            dold = self.desc[:]
            if callback != None and iflag == 1:
                callback(x, alphak)
            if self.args != None:
                fval = self.func(x, *self.args)
                gval = self.jac(x, *self.args)
            else:
                fval = self.func(x)
                gval = self.jac(x)
            gold_old = gold[:]
            gold = gval[:]
            if iflag == 1:
                icall += 1
            if iflag == 2:
                gnorm = np.sqrt(np.dot(gval, gval))
                xnorm = np.sqrt(np.dot(x,x))
                xnorm = max(1.0,xnorm)
                if gnorm/xnorm <= self.eps:
                    finish = True
                #tlev = self.eps*(1.0+np.abs(fval))
                #i = 0
                #if (np.abs(gval[i]) > tlev):
                #    continue
                #else:
                #    i += 1
                #if i >= self.n-1:
                #    finish = True
            if iflag <= 0:
                break
        if iflag == 0:
            if callback != None:
                callback(x, alphak)
            logger.info("minimization success")
            logger.info("iteration = {}".format(icall))
#                print("iteration = {}".format(icall))
            logger.info("final function value = {}".format(fval))
#                print("final function value = {}".format(fval))
            logger.info("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
#                print("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
            
        if iflag < 0:
            if callback != None:
                callback(x, alphak)
            logger.info("minimization failed, FLAG = {}".format(iflag))
            logger.info("iteration = {}".format(icall))
#                print("iteration = {}".format(icall))
            logger.info("final function value = {}".format(fval))
#                print("final function value = {}".format(fval))
            logger.info("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
#                print("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
            
        if iflag > 0:
            logger.info("minimization not converged")
#            print("minimization not converged")
            logger.info("current function value = {}".format(fval))
#            print("current function value = {}".format(fval))
            logger.info("current gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
#            print("current gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))

        return x, iflag

    def minimize_scipy(self, x0, callback=None):
        if self.method == "Nelder-Mead":
            if self.args is not None:
                res = spo.minimize(self.func, x0, args=self.args, method=self.method, \
                   options={'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            else:
                res = spo.minimize(self.func, x0, method=self.method, \
                   options={'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            logger.info("success={} message={}".format(res.success, res.message))
            logger.info("J={:7.3e} nit={}".format(res.fun, res.nit))
        elif self.method == "Powell":
            if self.args is not None:
                res = spo.minimize(self.func, x0, args=self.args, method=self.method, \
                   bounds=None, \
                   options={'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            else:
                res = spo.minimize(self.func, x0, method=self.method, \
                   bounds=None, \
                   options={'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            logger.info("success={} message={}".format(res.success, res.message))
            logger.info("J={:7.3e} nit={}".format(res.fun, res.nit))
        elif self.method == "dogleg" or self.method == "Newton-CG":
            if self.args is not None:
                res = spo.minimize(self.func, x0, args=self.args, method=self.method, \
                   jac=self.jac, hess=self.hess, options={'gtol':self.gtol, 'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            else:
                res = spo.minimize(self.func, x0, method=self.method, \
                   jac=self.jac, hess=self.hess, options={'gtol':self.gtol, 'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            logger.info("success={} message={}".format(res.success, res.message))
            logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
                res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
        else:
            if self.args is not None:
                if self.jac is None:
                    res = spo.minimize(self.func, x0, args=self.args, method=self.method, \
                        jac='2-point', options={'gtol':self.gtol, 'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
                else:
                    res = spo.minimize(self.func, x0, args=self.args, method=self.method, \
                        jac=self.jac, options={'gtol':self.gtol, 'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            else:
                if self.jac is None:
                    res = spo.minimize(self.func, x0, method=self.method, \
                        jac='2-point', options={'gtol':self.gtol, 'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
                else:
                    res = spo.minimize(self.func, x0, method=self.method, \
                        jac=self.jac, options={'gtol':self.gtol, 'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            logger.info("success={} message={}".format(res.success, res.message))
            logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
                res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))

        if res.success:
            iflag = 0
        else:
            iflag = -1

        return res.x, iflag

if __name__ == "__main__":        
    from scipy.optimize import rosen, rosen_der, rosen_hess
    import time

    n = 100
    iprint = np.ones(2, dtype=np.int32)
    iprint[0] = 0
    iprint[1] = 0
    print(iprint)

    args = None
    method = "LBFGS"
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
     method=method, maxiter=None)

    # initial guess
    x0 = np.zeros(n)
    for i in range(0, n, 2):
        x0[i] = -1.2
        x0[i+1] = 1.0
    
    start = time.time()
    #x = minimize.minimize_lbfgs(x0)
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")

    method = "BFGS"
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
     method=method, maxiter=None)
    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")

    method = "BFGS-jacfree"
    minimize = Minimize(n, rosen, jac=None, args=args, iprint=iprint,
     method="BFGS", maxiter=None)
    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")

    method = "CG"
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
     method=method, maxiter=None)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")

    method = "CG-jacfree"
    minimize = Minimize(n, rosen, jac=None, args=args, iprint=iprint,
     method="CG", maxiter=None)
    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")

    method = "Nelder-Mead"
    minimize = Minimize(n, rosen, args=args, iprint=iprint,
     method=method, maxiter=None)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")

    method = "Powell"
    minimize = Minimize(n, rosen, args=args, iprint=iprint,
     method=method, maxiter=None)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")

    method = "GD"
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
     method=method, maxiter=None)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")

    method = "CGF"
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
     method=method, maxiter=None, cgtype=3)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")

    method = "Newton-CG"
    minimize = Minimize(n, rosen, jac=rosen_der, hess=rosen_hess, args=args, iprint=iprint,
     method=method, maxiter=None)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")