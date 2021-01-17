import sys
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
obs_s = float(sys.argv[4])
oberr = str(int(obs_s*1e4)).zfill(4)
if model == "z08" or model == "z05":
    nx = 81
    perts = ["etkf-jh", "etkf-fh"]
    #na = 20
elif model == "l96":
    nx = 40
    perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
    #na = 300
x = np.arange(nx+1)
lx = np.arange(21) + 1
cmap = "coolwarm"
#plt.rcParams['axes.labelsize'] = 16 # fontsize arrange
fig, ax = plt.subplots(2)
for j in range(2):
    pt = perts[j]
    f = "{}_dhdx_{}_{}_cycle{}_oberr{}.npy".format(model, op, pt, 0, oberr)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        exit
    dhdx = np.load(f)
    ymin = np.min(dhdx)*1.001
    ymax = np.max(dhdx)*1.001
    ylim = max(np.abs(ymin),ymax)
    mappable0 = ax[j].pcolor(x,x,dhdx,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
    ax[j].set_aspect("equal")
    ax[j].set_xticks(x[::10])
    ax[j].set_yticks(x[::10])
    ax[j].set_ylabel("grid point")
    ax[j].set_xlabel("grid point")
    ax[j].set_title(pt)
    pp = fig.colorbar(mappable0, ax=ax[j], orientation="vertical")

fig.tight_layout()
fig.savefig("{}_dhdx_{}_oberr{}.png".format(model,op,oberr))