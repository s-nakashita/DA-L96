import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
if model == "z08" or model == "z05":
    nx = 81
    #na = 20
elif model == "l96":
    nx = 40
    #na = 100
ut = np.load("{}_ut.npy".format(model))[na-4:na,:]
perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
pa = np.zeros((nx, nx))
x = np.arange(nx+1) + 1
cmap = "Reds"
for pt in perts:
    dPa = np.load("{}_ua_{}_{}.npy".format(model, op, pt))[na-4:na,:,:] - ut[:,:,None]
    nmem = dPa.shape[2]
    fig, ax = plt.subplots(2,2)
    for j in range(2):
        for i in range(2):
            icycle = 2*j+i
            pa = dPa[icycle]@dPa[icycle].T / nmem
            
            ax[j, i].pcolor(x,x,pa,cmap=cmap)
            ax[j, i].set_aspect("equal")
            ax[j, i].set_xticks(x[::5])
            ax[j, i].set_yticks(x[::5])
            ax[j, i].invert_xaxis()
            ax[j, i].invert_yaxis()
            ax[j, i].set_title("cycle{}".format(na-3+icycle))
    axpos = ax[0, 1].get_position()
    cbar_ax = fig.add_axes([0.90, axpos.y0/4, 0.02, 2*axpos.height])
    mappable = ScalarMappable(cmap=cmap)
    fig.colorbar(mappable, cax=cbar_ax)
    fig.tight_layout()
    fig.savefig("{}_pat_{}_{}.png".format(model,op,pt))
