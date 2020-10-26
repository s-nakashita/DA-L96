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
    #na = 300
perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
pa = np.zeros((nx, nx))
x = np.arange(nx+1) + 1
cmap = "Reds"
for pt in perts:
    if pt == "mlef" or pt == "grad":
        dPa = np.load("{}_pa_{}_{}.npy".format(model, op, pt))[na-4:na,:,:]
        nmem = dPa.shape[2]
    else:
        Pa = np.load("{}_pa_{}_{}.npy".format(model, op, pt))[na-4:na,:,:]
    fig, ax = plt.subplots(2,2)
    for j in range(2):
        for i in range(2):
            icycle = 2*j+i
            if pt == "mlef" or pt == "grad":
                pa = dPa[icycle]@dPa[icycle].T / (nmem-1)
            else:
                pa = Pa[icycle]
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
    fig.savefig("{}_pa_{}_{}.png".format(model,op,pt))
