import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
if model == "z08" or model == "z05":
    nx = 81
elif model == "l96":
    nx = 40
linecolor={"mlef":"tab:blue","grad":"tab:orange"}
linestyle={"jb":"dashdot","jo":"dashed"}
for k in range(4):
    f = "{}_cJ_{}_{}_cycle{}.npy".format(model, op, "mlef", k)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    cJ_mlef = np.load(f)
    f = "{}_cJ_{}_{}_cycle{}.npy".format(model, op, "grad", k)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    cJ_grad = np.load(f)
    #cJb_mlef = np.load("{}_cJb_{}_{}_cycle{}.npy".format(model, op, "mlef", k))
    #cJb_grad = np.load("{}_cJb_{}_{}_cycle{}.npy".format(model, op, "grad", k))
    #cJo_mlef = np.load("{}_cJo_{}_{}_cycle{}.npy".format(model, op, "mlef", k))
    #cJo_grad = np.load("{}_cJo_{}_{}_cycle{}.npy".format(model, op, "grad", k))
    xopt_mlef = cJ_mlef[0]
    xopt_grad = cJ_grad[0]
    #xopt_mlef = cJb_mlef[0]
    #xopt_grad = cJb_grad[0]
    lenx = cJ_mlef.shape[0]-1
    maxis = np.linspace(-lenx//4,lenx//4,lenx)
    lenx = cJ_grad.shape[0]-1
    gaxis = np.linspace(-lenx//4,lenx//4,lenx)
    fig, ax = plt.subplots(1,4,figsize=(8,2))
    #,figsize=(6, 1))
    fig.subplots_adjust(top=0.9,bottom=0.1)
    #fig = plt.figure(figsize=(6, 1))
    #gs = gridspec.GridSpec(nrows=1, ncols=4)
    for j in range(2):
        for i in range(2):
            #ax = fig.add_subplot(1, 4, 2*j+i+1)
            xm = xopt_mlef[2*j+i]
            ym = getopt(maxis, xm, cJ_mlef[1:, 2*j+i])
            #ymb = getopt(maxis, xm, cJb_mlef[1:, 2*j+i])
            #ymo = getopt(maxis, xm, cJo_mlef[1:, 2*j+i])
            #ym = ymb + ymo
            xg = xopt_grad[2*j+i]
            yg = getopt(gaxis, xg, cJ_grad[1:, 2*j+i])
            #ygb = getopt(gaxis, xg, cJb_grad[1:, 2*j+i])
            #ygo = getopt(gaxis, xg, cJo_grad[1:, 2*j+i])
            #yg = ygb + ygo
            ax[2*j+i].plot(maxis, cJ_mlef[1:, 2*j+i], color=linecolor["mlef"], label="mlef")
            ax[2*j+i].plot(gaxis, cJ_grad[1:, 2*j+i], color=linecolor["grad"], label="grad")
            #ax[j, i].plot(maxis, cJb_mlef[1:, 2*j+i], color=linecolor["mlef"],
            # linestyle=linestyle["jb"], label="mlef-jb")
            #ax[j, i].plot(gaxis, cJb_grad[1:, 2*j+i], color=linecolor["grad"], 
            # linestyle=linestyle["jb"], label="grad-jb")
            #ax[j, i].plot(maxis, cJo_mlef[1:, 2*j+i], color=linecolor["mlef"],
            # linestyle=linestyle["jo"], label="mlef-jo")
            #ax[j, i].plot(gaxis, cJo_grad[1:, 2*j+i], color=linecolor["grad"], 
            # linestyle=linestyle["jo"], label="grad-jo")
            ax[2*j+i].plot(xm, ym, marker='o', linestyle="none", color=linecolor["mlef"])
            ax[2*j+i].plot(xg, yg, marker='^', linestyle="none", color=linecolor["grad"])
            #if op == "cubic" and k==0:
            #    ax[j, i].set_xlim(-100.0, 100.0)
            #print("mlef-jb max {:7.3e}".format(np.max(cJb_mlef[1:, 2*j+i])))
            #print("grad-jb max {:7.3e}".format(np.max(cJb_grad[1:, 2*j+i])))
            #print("mlef-jo max {:7.3e}".format(np.max(cJo_mlef[1:, 2*j+i])))
            #print("grad-jo max {:7.3e}".format(np.max(cJo_grad[1:, 2*j+i])))
            mmax = np.max(cJ_mlef[1:, 2*j+i])
            mmin = np.min(cJ_mlef[1:, 2*j+i])
            gmax = np.max(cJ_grad[1:, 2*j+i])
            gmin = np.min(cJ_grad[1:, 2*j+i])
            #mmax = max(np.max(cJb_mlef[1:, 2*j+i]),np.max(cJo_mlef[1:, 2*j+i]))
            #mmin = min(np.min(cJb_mlef[1:, 2*j+i]),np.min(cJo_mlef[1:, 2*j+i]))
            #gmax = max(np.max(cJb_grad[1:, 2*j+i]),np.max(cJo_grad[1:, 2*j+i]))
            #gmin = min(np.min(cJb_grad[1:, 2*j+i]),np.min(cJo_grad[1:, 2*j+i]))
            ymin = 10**(int(np.log10(min(mmin,gmin)))-1)
            print(ymin)
            if gmax > 5e6 or mmax > 5e6:
                ax[2*j+i].set_ylim(-1e6,5e6)
            #ax[2*j+i].set_yticks(fontsize=8)
            #ax[2*j+i].set_xticks(x[::200],fontsize=8)
            #ax[2*j+i].set_xticks(x[::100], minor=True)
            ax[2*j+i].set_xlabel("zeta",fontsize=8)
            ax[2*j+i].set_ylabel("J",fontsize=8)
            #ax[2*j+i].set_aspect("1")
            ax[2*j+i].set_title("direction {}".format(2*j+i+1),fontsize=9)
            if 2*j+i == 0:
                ax[2*j+i].legend(loc=9,fontsize=8)
    fig.suptitle("{} cycle{}".format(op,k+1))
    plt.tight_layout(pad=1) 
    #fig.tight_layout() 
    #(rect=[0,0,1,0.96])
    fig.savefig("{}_cJ_{}_cycle{}.png".format(model, op, k))
    fig.savefig("{}_cJ_{}_cycle{}.pdf".format(model, op, k))
    #fig.savefig("{}_cJb+o_{}_cycle{}.png".format(model, op, k))
