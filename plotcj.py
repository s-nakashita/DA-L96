import sys
import numpy as np
import matplotlib.pyplot as plt

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
    print(x1, xopt, x2)
    return yopt

def getNearestValue(arr, num):
    idx = np.abs(arr - num).argmin()
    return arr[idx], idx

op = sys.argv[1]
model = sys.argv[2]
if model == "z08" or model == "z05":
    nx = 81
elif model == "l96":
    nx = 40
for k in range(4):
    cJ_mlef = np.load("{}_cJ_{}_{}_cycle{}.npy".format(model, op, "mlef", k))
    cJ_grad = np.load("{}_cJ_{}_{}_cycle{}.npy".format(model, op, "grad", k))
    xopt_mlef = cJ_mlef[0]
    xopt_grad = cJ_grad[0]
    lenx = cJ_mlef.shape[0]-1
    x = np.linspace(-lenx//4,lenx//4,lenx)
    fig, ax = plt.subplots(2,2)
    for j in range(2):
        for i in range(2):
            xm = xopt_mlef[2*j+i]
            ym = getopt(x, xm, cJ_mlef[1:, 2*j+i])
            xg = xopt_grad[2*j+i]
            yg = getopt(x, xg, cJ_grad[1:, 2*j+i])
            ax[j, i].plot(x, cJ_mlef[1:, 2*j+i], label="mlef")
            ax[j, i].plot(x, cJ_grad[1:, 2*j+i], linestyle="--", label="grad")
            ax[j, i].plot(xm, ym, marker='o', markerfacecolor="none", label="mlef")
            ax[j, i].plot(xg, yg, marker='^', markerfacecolor="none", label="grad")
            #if op == "cubic" and k==0:
            #    ax[j, i].set_xlim(-100.0, 100.0)
            #ax[j, i].set_ylim(-1e5,1e7)
            #ax[j, i].set_xticks(x[::200])
            #ax[j, i].set_xticks(x[::100], minor=True)
            ax[j, i].set_title("direction {}".format(2*j+i+1))
    ax[0, 0].legend()
    fig.suptitle("{} cycle{}".format(op,k+1))
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig("{}_cJ_{}_cycle{}.png".format(model, op, k))
