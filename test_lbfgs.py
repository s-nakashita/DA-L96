import lbfgs
from scipy.optimize import rosen, rosen_der
import numpy as np

print(lbfgs.__doc__)
print(lbfgs.lbfgs.__doc__)
n = 100
m = 5
iprint = np.ones(2, dtype=np.int32)
iprint[0] = 1
iprint[1] = 0
print(iprint)

lwork = n*(2*m+1)+2*m
work = np.zeros(lwork)

diagco = False
diag = np.ones(n)
eps = 1.0e-5
xtol = 1.0e-16
icall = 0
iflag = 0

# initial guess
x = np.zeros(n)
for i in range(0, n, 2):
    x[i] = -1.2
    x[i+1] = 1.0
x0 = x.copy()
# initial evaluation
fval = 0.0
gval = np.zeros_like(x)
fval = rosen(x)
gval = rosen_der(x)
print("initial function value = {}".format(fval))
print("initial gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))

for icall in range(2000):
    [xk, oflag] = lbfgs.lbfgs(n=n, m=m, x=x, f=fval, g=gval, \
        diagco=diagco, diag=diag, \
        iprint=iprint, eps=eps, xtol=xtol, w=work, iflag=iflag)
    iflag = oflag
    x = xk[:]
    fval = rosen(x)
    gval = rosen_der(x)
    print("iflag = {}".format(iflag))
    #print("x - x0 = {}".format(x-x0))
    #print("diag = {}".format(diag))
    #print("current function value = {}".format(fval))
    #print("current gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
    if iflag <= 0:
        break
print("iteration number = {}".format(icall))
print("x = {}".format(x))
print("final function value = {}".format(fval))
print("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))