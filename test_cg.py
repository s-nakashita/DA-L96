import cg
from scipy.optimize import rosen, rosen_der
import numpy as np

print(cg.__doc__)
print(cg.cgfam.__doc__)
n = 2
method = 2
irest = 1
iprint = np.ones(2, dtype=np.int32)
iprint[0] = 1
iprint[1] = 0
print(iprint)

desc = np.ones(n)
dold = desc.copy()
work = np.zeros(n)
eps = 1.0e-5
xtol = 1.0e-16
icall = 0
iflag = 0
finish = False

# initial guess
x = np.zeros(n)
for i in range(n):
    x[i] = -2.0
x0 = x.copy()
xold = x.copy()
# initial evaluation
fval = 0.0
gval = np.zeros_like(x)
fval = rosen(x)
gval = rosen_der(x)
gold = gval.copy()
gold_old = gval.copy()
print("initial function value = {}".format(fval))
print("initial gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
xold = x[:]
gold = gval[:]
gold_old = gold[:]
dold = desc[:]    
while icall < 2000:
    print("finish = {}".format(finish))
    [x,gval,desc,gold,oflag,ofinish] = \
        cg.cgfam(n=n, x=xold, f=fval, g=gold, \
        d=dold, gold=gold_old,\
        iprint=iprint, eps=eps, w=work, iflag=iflag, \
        irest=irest, method=method, finish=finish)
    iflag = oflag
    finish = bool(ofinish==1)
    xold = x[:]
    dold = desc[:]
    fval = rosen(x)
    gval = rosen_der(x)
    gold_old = gold[:]
    gold = gval[:]
    print("iflag = {}".format(iflag))
    if iflag == 1:
        icall += 1
    #print("x - x0 = {}".format(x-x0))
    #print("diag = {}".format(diag))
    #print("current function value = {}".format(fval))
    #print("current gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
    if iflag == 2:
        tlev = eps*(1.0+np.abs(fval))
        i = 0
        if (np.abs(gval[i]) > tlev):
            print(f"g[{i}] > f")
            continue
        else:
            i += 1
        if i >= n-1:
            finish = True
    if iflag <= 0:
        break
print("iteration number = {}".format(icall))
print("x = {}".format(x))
print("final function value = {}".format(fval))
print("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))