import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker, colors

def getopt(x, xopt, y):
    x1, id1 = getNearestValue(x, xopt)
    if x1 < xopt:
        id2 = id1+1
    else:
        id2 = id1-1
    if id2 < 0:
        id2 = 0
    if id2 > len(x) - 1:
        id2 = len(x) - 1
    x2 = x[id2]
    yopt = 0.5*(y[id1] + y[id2])
    #print(x1, xopt, x2)
    return yopt

def getNearestValue(arr, num):
    idx = np.abs(arr - num).argmin()
    return arr[idx], idx

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = sys.argv[4]
if model == "z08" or model == "z05":
    nx = 81
elif model == "l96":
    nx = 40
linecolor={"mlef":"tab:blue","grad":"tab:orange"}
linestyle={"jb":"dashdot","jo":"dashed"}
f = "{}_x+g_{}_{}_cycle0.npy".format(model, op, pt)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit()
x_g = np.load(f)
g = x_g[0,:]
gap = x_g[1,:]
xmax = x_g[2,0]
xk = x_g[3:,:]
nmem = x_g.shape[1]
for i in range(nmem-1):
    for j in range(i+1,nmem):
        f = "{}_cJ2d_{}_{}_{}{}_cycle0.npy".format(model, op, pt, i, j)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue

        cJ = np.load(f)
        diff = np.log10(cJ.max())-np.log10(cJ.min())
        print(diff)
        nx = cJ.shape[0]
        xaxis = np.linspace(-xmax, xmax, cJ.shape[0], endpoint=True)
        yaxis = xaxis.copy()
        xx, yy = np.meshgrid(xaxis, yaxis)
        fig, ax = plt.subplots()
        if diff > 2:
            lev_exp = np.arange(np.floor(np.log10(cJ.min())-1),
                       np.ceil(np.log10(cJ.max())+1))
            print(lev_exp)
            levs = np.zeros(len(lev_exp)*2-1)
            for k in range(len(lev_exp)-1):
                levs[2*k] = np.power(10, lev_exp[k])
                levs[2*k+1] = 5.0*np.power(10, lev_exp[k])
            levs[-1] = np.power(10, lev_exp[-1])
            print(levs)
        #ax.contour(xx, yy, cJ, levs)
            cntr = ax.contourf(xx, yy, cJ, levs, norm=colors.LogNorm())
        else:
            cntr = ax.contourf(xx, yy, cJ)
        #ax.contour(xx, yy, cJ, levels=8, locator=ticker.LogLocator())
        #cntr = ax.contourf(xx, yy, cJ, levels=8, locator=ticker.LogLocator())
        for k in range(xk.shape[0]-1):
            #ax.text(xk[k,i], xk[k,j], k, ha="center", color="k", size=14)
            ax.scatter(xk[k,i], xk[k,j], marker='^', color='orange')
        ax.scatter(xk[-1,i], xk[-1,j], marker='x', color='red')
        ax.quiver(0.0, 0.0, -g[i], -g[j], color='red')
        #ax.quiver(0.0, 0.0, -gap[i], -gap[j], color='pink')
        ax.set_xticks(xaxis[::10])
        ax.set_yticks(yaxis[::10])
        ax.set_xlabel(f'zeta{i+1}')
        ax.set_ylabel(f'zeta{j+1}')
        ax.set_aspect("equal")
        ax.set_title(f"zeta{i+1}-zeta{j+1}")
        fig.colorbar(cntr, ax=ax)
        fig.savefig("{}_cJ2d_{}_{}_{}{}_cycle0.png".format(model, op, pt, i, j))
        