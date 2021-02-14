import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
if model == "z08" or model == "z05":
    nx = 81
    perts = ["etkf-jh", "etkf-fh"]
    #na = 20
elif model == "l96":
    nx = 40
    perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
    #na = 300
x = np.arange(21) + 1
y = np.arange(31) + 1
cmap = "coolwarm"
for pt in perts:
    fig, ax = plt.subplots(2,1)
    f = "{}_K_{}_{}_cycle{}_wL.npy".format(model, op, pt, 0)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    K = np.load(f)[:20,:30]
    j=0
    ymin = np.min(K)-0.1
    ymax = np.max(K)+0.1
    ylim = max(np.abs(ymin),ymax)
    mappable0 = ax[j].pcolor(y,x,K,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
    ax[j].set_aspect("equal")
    ax[j].set_xticks(x[::5])
    ax[j].set_yticks(x[::5])
            #ax[j, i].invert_xaxis()
            #ax[j, i].invert_yaxis()
    ax[j].set_title("Before")
    pp = fig.colorbar(mappable0, ax=ax[j], orientation="vertical")
    #pp.set_clim(ymin,ymax)
    f = "{}_Kloc_{}_{}_cycle{}_wL.npy".format(model, op, pt, 0)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    Kloc = np.load(f)[:20,:20]
    print(np.min(Kloc),np.max(Kloc))
    j=1
    ymin = np.min(Kloc)-0.1
    ymax = np.max(Kloc)+0.1
    ylim = max(np.abs(ymin),ymax)
    mappable1 = ax[j].pcolor(x,x,Kloc,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
    ax[j].set_aspect("equal")
    ax[j].set_xticks(x[::5])
    ax[j].set_yticks(x[::5])
            #ax[j, i].invert_xaxis()
            #ax[j, i].invert_yaxis()
    ax[j].set_title("After")
    #axpos = ax[0].get_position()
    #cbar_ax = fig.add_axes([0.90, axpos.y0/4, 0.02, 2*axpos.height])
    #mappable = ScalarMappable(cmap=cmap)
    pp = fig.colorbar(mappable1, ax=ax[j], orientation="vertical")
    #pp.set_clim(ymin,ymax)
    fig.tight_layout()
    fig.savefig("testk.png")
