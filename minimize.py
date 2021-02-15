from lbfgs import lbfgs
from scipy.optimize import rosen, rosen_der
import numpy as np

class Minimize():
    def __init__(self, n, m, func, jac, args, iprint):
        self.n = n
        self.m = m
        self.func = func
        self.jac = jac
        self.args = args
        self.iprint = iprint
        self.lwork = self.n*(2*self.m+1)+2*self.m
        self.work = np.zeros(self.lwork)
        self.eps = 1.0e-5
        self.xtol = 1.0e-16
        self.diagco = False
        self.diag = np.ones(self.n)

    def minimize_lbfgs(self, x0):
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
        print("initial function value = {}".format(fval))
        print("initial gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))

        x = x0.copy()
        for icall in range(1000):
            oflag = lbfgs(n=self.n, m=self.m, x=x, f=fval, g=gval, \
                          diagco=self.diagco, diag=self.diag, \
                          iprint=self.iprint, eps=self.eps, xtol=self.xtol, w=self.work, iflag=iflag)
            iflag = oflag
            if self.args != None:
                fval = self.func(x, *self.args)
                gval = self.jac(x, *self.args)
            else:
                fval = self.func(x)
                gval = self.jac(x)
            if iflag <= 0:
                print("iteration = {}".format(icall))
                print("final function value = {}".format(fval))
                print("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
                break
        if iflag > 0:
            print("minimization not converged")
            print("current function value = {}".format(fval))
            print("current gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))

        return x    

if __name__ == "__main__":        
    n = 100
    m = 7
    iprint = np.ones(2, dtype=np.int32)
    iprint[0] = 1
    iprint[1] = 0
    print(iprint)

    args = None
    minimize = Minimize(n, m, rosen, rosen_der, args, iprint)

    # initial guess
    x0 = np.zeros(n)
    for i in range(0, n, 2):
        x0[i] = -1.2
        x0[i+1] = 1.0
    
    x = minimize.minimize_lbfgs(x0)
    print(x)