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
obs_s = 0.005
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
fig, ax = plt.subplots(2,2)
pt = perts[0]
f = "{}_K2_{}_{}_cycle{}_oberr{}.npy".format(model, op, pt, 0, oberr)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
K2 = np.load(f)
HPH = K2 - np.diag(np.ones(nx)*obs_s*obs_s)
ymin = np.min(HPH)*1.001
ymax = np.max(HPH)*1.001
ylim = max(np.abs(ymin),ymax)
mappable0 = ax[0,0].pcolor(x,x,HPH,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
ax[0,0].set_aspect("equal")
ax[0,0].set_xticks(x[::10])
ax[0,0].set_yticks(x[::10])
ax[0,0].set_ylabel("observation point")
ax[0,0].set_xlabel("observation point")
ax[0,0].set_title(pt)
pp = fig.colorbar(mappable0, ax=ax[0,0], orientation="vertical")
diff = HPH

pt = perts[1]
f = "{}_K2_{}_{}_cycle{}_oberr{}.npy".format(model, op, pt, 0, oberr)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
K2 = np.load(f)
HPH = K2 - np.diag(np.ones(nx)*obs_s*obs_s)
ymin = np.min(HPH)*1.001
ymax = np.max(HPH)*1.001
ylim = max(np.abs(ymin),ymax)
mappable0 = ax[0,1].pcolor(x,x,HPH,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
ax[0,1].set_aspect("equal")
ax[0,1].set_xticks(x[::10])
ax[0,1].set_yticks(x[::10])
ax[0,1].set_ylabel("observation point")
ax[0,1].set_xlabel("observation point")
ax[0,1].set_title(pt)
pp = fig.colorbar(mappable0, ax=ax[0,1], orientation="vertical")

diff = diff - HPH
ymin = np.min(diff)*1.001
ymax = np.max(diff)*1.001
ylim = max(np.abs(ymin),ymax)
mappable0 = ax[1,0].pcolor(x,x,diff,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
ax[1,0].set_aspect("equal")
ax[1,0].set_xticks(x[::10])
ax[1,0].set_yticks(x[::10])
ax[1,0].set_ylabel("observation point")
ax[1,0].set_xlabel("observation point")
ax[1,0].set_title(perts[0]+"-"+perts[1])
pp = fig.colorbar(mappable0, ax=ax[1,0], orientation="vertical")

fig.tight_layout()
fig.savefig("{}_hph_{}_oberr{}.png".format(model,op,oberr))