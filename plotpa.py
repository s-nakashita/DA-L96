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
    perts = ["mlef", "grad", "etkf-jh", "etkf-fh"]
    cycle = [1, 2, 3, 4]
    #na = 20
elif model == "l96":
    nx = 40
    perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
    #na = 300
nx = 30
pa = np.zeros((nx, nx))
x = np.arange(nx+1) + 1
cmap = "Reds"
for pt in perts:
    fig, ax = plt.subplots(2,2)
    for j in range(2):
        for i in range(2):
            icycle = cycle[2*j+i]
            if pt == "mlef" or pt == "grad":
                f = "{}_pa_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
                if not os.path.isfile(f):
                    print("not exist {}".format(f))
                    continue
                dPa = np.load(f)[:nx,:]
                #nmem = dPa.shape[2]
            else:
                f = "{}_pa_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
                if not os.path.isfile(f):
                    print("not exist {}".format(f))
                    continue
                Pa = np.load(f)[:nx,:nx]
    
            if pt == "mlef" or pt == "grad":
                pa = dPa@dPa.T# / (nmem-1)
            else:
                pa = Pa
            ymax = np.max(pa)
            ymin = np.min(pa)
            mappable=ax[j, i].pcolor(x,x,pa,cmap=cmap,norm=Normalize(vmin=ymin, vmax=ymax))
            ax[j, i].set_aspect("equal")
            ax[j, i].set_xticks(x[::5])
            ax[j, i].set_yticks(x[::5])
            #ax[j, i].invert_xaxis()
            #ax[j, i].invert_yaxis()
            ax[j, i].set_title("cycle{}".format(icycle+1))
            fig.colorbar(mappable, ax=ax[j,i],orientation="vertical")
    #axpos = ax[0, 1].get_position()
    #cbar_ax = fig.add_axes([0.90, axpos.y0/4, 0.02, 2*axpos.height])
    #mappable = ScalarMappable(cmap=cmap)
    #fig.colorbar(mappable, cax=cbar_ax)
    fig.tight_layout()
    fig.savefig("{}_pa_{}_{}.png".format(model,op,pt))
