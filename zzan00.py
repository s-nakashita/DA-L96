import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def f1(x):
    return 2 * x - 2


def f2(x):
    return x - 4


def f(x):
    dt = 0.1
    for i in range(4):
        if x < 1:
            x += f1(x) * dt
        else:
            x += f2(x) * dt
    return x ** 2


def g(x):
    dt = 0.1
    g = 1.0
    for i in range(4):
        if x < 1:
            x += f1(x) * dt
            g *= 1.0 + dt*2.0
        else:
            x += f2(x) * dt
            g *= 1.0 + dt*1.0        
    
    return 2 * x * g

def gf(dx, *args):
    x0, pf = args
    x = x0 + np.dot(pf, dx)
    return f(x)

def gg(dx, *args):
    x0, pf = args
    g = np.zeros_like(dx)
    #print(dx.shape)
    x = x0 + np.dot(pf, dx)
    for i in range(dx.size):
        g[i] = (f(x+pf[i]) - f(x))
    return g

def hess(x):
    return np.array([2])

def iter_opt(f, x0, method):
    x = []
    y = []
    niter = 30
    for i in range(niter):
        if method == "dogleg":
            result = optimize.minimize(f, x0, jac=g, hess=hess, method=method, 
            options={"maxiter": i}, tol=np.finfo(1.0).eps)
        elif method == "mlef":
            nmem = 10
            dx0 = np.zeros(nmem)
            pf = np.ones(nmem)/np.sqrt(nmem)
            args = x0, pf
            result = optimize.minimize(gf, dx0, jac=gg, args=args, method="BFGS",
                options={"maxiter":i}, tol=np.finfo(1.0).eps)
        else:
            result = optimize.minimize(f, x0, jac=g, method=method, 
                options={"maxiter": i}, tol=np.finfo(1.0).eps)
        
        if result.success:
            print(f"Converged at {i}")
            print(f"message={result.message}")
            print("J={:7.3e} dJ={:7.3e}".format( \
                result.fun, np.sqrt(result.jac.transpose() @ result.jac)))
            break
        else:
            print(f"not converged at {i} because {result.message}")
        if i == 0 and (method == "L-BFGS-B" or method == "mlef"):
            x.append(x0)
            y.append(f(x0))
        elif method == "mlef":
            xk = x0 + np.dot(pf, result.x)
            x.append(xk)
            y.append(f(xk))
        else:
            x.append(result.x)
            y.append(result.fun)
    return x, y, i


n = 351
m = 4
xbounds = -0.5, 3.0
method = "BFGS"
#method = "CG"
#method = "dogleg"
#method = "mlef"
#method = "L-BFGS-B"

if __name__ == "__main__":
    x = np.linspace(*xbounds, n)
    if len(sys.argv) > 1:
        method = sys.argv[1]
    #fig, ax = plt.subplots(1, 2)
    x0 = 2.0
    dx = 1.0e-1
    alpha = 1.0e-5
    diff1 = f(x0+alpha*dx) - f(x0)
    diff2 = alpha*g(x0)*dx
    print(diff1)
    print(diff2)
    print(diff1/diff2-1)

    fig, ax = plt.subplots()
    y = []
    yg = []
    for x0 in x:
        y.append(f(x0))
        yg.append(g(x0))
    ax.plot(x, yg)
    ax.set_xlabel("initial condition $x_0$")
    ax.set_ylabel(r"gradient of cost function $ \nabla J_1(x_0)$")
    #plt.show()
    #fig.savefig(f"zzan00_grad.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(x, y)
    x0 = 2.0
    xo, yo, io = iter_opt(f, x0, method)
    ax.scatter(xo[:m], yo[:m], color="k")
    ax.scatter(xo[-1], yo[-1], color="r", marker="x")
    for i in range(io-1):
        xloc = xo[i]
        yloc = yo[i] + (0.3 - i % 2)
        ax.text(xloc, yloc, i, ha="center", color="k", size=14)
    xloc = xo[-1]
    yloc = yo[-1] - 0.4
    ax.text(xloc, yloc, io, ha="center", color="r", size=14)
    ax.set_title(method)
    ax.set_xlabel("initial condition $x_0$")
    ax.set_ylabel("cost function $J_1(x_0)$")
    #plt.show()
    fig.savefig(f"zzan00_{method}.png")

