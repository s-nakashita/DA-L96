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
    #print(x1, xopt, x2)
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
linecolor={"mlef":"tab:blue","grad":"tab:orange"}
linestyle={"jb":"dashdot","jo":"dashed"}
for k in range(4):
    cJb_mlef = np.load("{}_cJb_{}_{}_cycle{}.npy".format(model, op, "mlef", k))
    cJb_grad = np.load("{}_cJb_{}_{}_cycle{}.npy".format(model, op, "grad", k))
    xopt_mlef = cJb_mlef[0]
    xopt_grad = cJb_grad[0]
    lenx = cJb_mlef.shape[0]-1
    maxis = np.linspace(-lenx//4,lenx//4,lenx)
    lenx = cJb_grad.shape[0]-1
    gaxis = np.linspace(-lenx//4,lenx//4,lenx)
    fig, ax = plt.subplots(2,2)
    for j in range(2):
        for i in range(2):
            xm = xopt_mlef[2*j+i]
            ym = getopt(maxis, xm, cJb_mlef[1:, 2*j+i])
            xg = xopt_grad[2*j+i]
            yg = getopt(gaxis, xg, cJb_grad[1:, 2*j+i])
            ax[j, i].plot(maxis, cJb_mlef[1:, 2*j+i], color=linecolor["mlef"],
             label="mlef-jb")
            ax[j, i].plot(gaxis, cJb_grad[1:, 2*j+i], color=linecolor["grad"], 
             label="grad-jb")
            ax[j, i].plot(xm, ym, marker='o', linestyle="none", color=linecolor["mlef"])
            ax[j, i].plot(xg, yg, marker='^', linestyle="none", color=linecolor["grad"])
            #if op == "cubic" and k==0:
            #    ax[j, i].set_xlim(-100.0, 100.0)
            print("mlef-jb max {:7.3e}".format(np.max(cJb_mlef[1:, 2*j+i])))
            print("grad-jb max {:7.3e}".format(np.max(cJb_grad[1:, 2*j+i])))
            mmax = np.max(cJb_mlef[1:, 2*j+i])
            mmin = np.min(cJb_mlef[1:, 2*j+i])
            gmax = np.max(cJb_grad[1:, 2*j+i])
            gmin = np.min(cJb_grad[1:, 2*j+i])
            ymin = 10**(int(np.log10(min(mmin,gmin)))-1)
            print(ymin)
            if gmax > 5e6 or mmax > 5e6:
                ax[j, i].set_ylim(-1e6,5e6)
            #ax[j, i].set_xticks(x[::200])
            #ax[j, i].set_xticks(x[::100], minor=True)
            ax[j, i].set_xlabel("zeta")
            ax[j, i].set_ylabel("J")
            ax[j, i].set_title("direction {}".format(2*j+i+1))
    ax[0, 0].legend(loc=9)
    fig.suptitle("{} cycle{}".format(op,k+1))
    fig.tight_layout(rect=[0,0,1,0.96])
    #fig.savefig("{}_cJ_{}_cycle{}.png".format(model, op, k))
    fig.savefig("{}_cJb_{}_cycle{}.png".format(model, op, k))
for k in range(4):
    cJo_mlef = np.load("{}_cJo_{}_{}_cycle{}.npy".format(model, op, "mlef", k))
    cJo_grad = np.load("{}_cJo_{}_{}_cycle{}.npy".format(model, op, "grad", k))
    xopt_mlef = cJo_mlef[0]
    xopt_grad = cJo_grad[0]
    lenx = cJo_mlef.shape[0]-1
    maxis = np.linspace(-lenx//4,lenx//4,lenx)
    lenx = cJo_grad.shape[0]-1
    gaxis = np.linspace(-lenx//4,lenx//4,lenx)
    fig, ax = plt.subplots(2,2)
    for j in range(2):
        for i in range(2):
            xm = xopt_mlef[2*j+i]
            ym = getopt(maxis, xm, cJo_mlef[1:, 2*j+i])
            xg = xopt_grad[2*j+i]
            yg = getopt(gaxis, xg, cJo_grad[1:, 2*j+i])
            ax[j, i].plot(maxis, cJo_mlef[1:, 2*j+i], color=linecolor["mlef"],
             label="mlef-jo")
            ax[j, i].plot(gaxis, cJo_grad[1:, 2*j+i], color=linecolor["grad"], 
             label="grad-jo")
            ax[j, i].plot(xm, ym, marker='o', linestyle="none", color=linecolor["mlef"])
            ax[j, i].plot(xg, yg, marker='^', linestyle="none", color=linecolor["grad"])
            #if op == "cubic" and k==0:
            #    ax[j, i].set_xlim(-100.0, 100.0)
            print("mlef-jo max {:7.3e}".format(np.max(cJo_mlef[1:, 2*j+i])))
            print("grad-jo max {:7.3e}".format(np.max(cJo_grad[1:, 2*j+i])))
            mmax = np.max(cJo_mlef[1:, 2*j+i])
            mmin = np.min(cJo_mlef[1:, 2*j+i])
            gmax = np.max(cJo_grad[1:, 2*j+i])
            gmin = np.min(cJo_grad[1:, 2*j+i])
            ymin = 10**(int(np.log10(min(mmin,gmin)))-1)
            print(ymin)
            #if gmax > 5e6 or mmax > 5e6:
            #    ax[j, i].set_ylim(ymin,5e6)
            #ax[j, i].set_xticks(x[::200])
            #ax[j, i].set_xticks(x[::100], minor=True)
            ax[j, i].set_xlabel("zeta")
            ax[j, i].set_ylabel("J")
            ax[j, i].set_title("direction {}".format(2*j+i+1))
    ax[0, 0].legend(loc=9)
    fig.suptitle("{} cycle{}".format(op,k+1))
    fig.tight_layout(rect=[0,0,1,0.96])
    #fig.savefig("{}_cJ_{}_cycle{}.png".format(model, op, k))
    fig.savefig("{}_cJo_{}_cycle{}.png".format(model, op, k))