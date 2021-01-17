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
cmap = "coolwarm"
#plt.rcParams['axes.labelsize'] = 16 # fontsize arrange
fig, ax = plt.subplots(2,2)
ff = "{}_K1_{}_etkf-fh_cycle{}.npy".format(model, op, 0)
if not os.path.isfile(ff):
    print("not exist {}".format(ff))
    exit
Kf = np.load(ff)[:20,:20]
fj = "{}_K1_{}_etkf-jh_cycle{}.npy".format(model, op, 0)
if not os.path.isfile(fj):
    print("not exist {}".format(fj))
    exit
Kj = np.load(fj)[:20,:20]
K = Kf - Kj
ymin = np.min(K)#-0.1
ymax = np.max(K)#+0.1
ylim = max(np.abs(ymin),ymax)
mappable0 = ax[0,0].pcolor(x,x,K,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
ax[0,0].set_aspect("equal")
ax[0,0].set_xticks(x[::5])
ax[0,0].set_yticks(x[::5])
ax[0,0].set_ylabel("grid point")
ax[0,0].set_xlabel("observation point")
ax[0,0].set_title("K1")
pp = fig.colorbar(mappable0, ax=ax[0,0], orientation="vertical")

ff = "{}_K2_{}_etkf-fh_cycle{}.npy".format(model, op, 0)
if not os.path.isfile(ff):
    print("not exist {}".format(ff))
    exit
Kf = np.load(ff)[:20,:20]
fj = "{}_K2_{}_etkf-jh_cycle{}.npy".format(model, op, 0)
if not os.path.isfile(fj):
    print("not exist {}".format(fj))
    exit
Kj = np.load(fj)[:20,:20]
K = Kf - Kj
ymin = np.min(K)#-0.1
ymax = np.max(K)#+0.1
ylim = max(np.abs(ymin),ymax)
mappable0 = ax[1,0].pcolor(x,x,K,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
ax[1,0].set_aspect("equal")
ax[1,0].set_xticks(x[::5])
ax[1,0].set_yticks(x[::5])
ax[1,0].set_ylabel("grid point")
ax[1,0].set_xlabel("observation point")
ax[1,0].set_title("K2")
pp = fig.colorbar(mappable0, ax=ax[1,0], orientation="vertical")

ff = "{}_K_{}_etkf-fh_cycle{}.npy".format(model, op, 0)
if not os.path.isfile(ff):
    print("not exist {}".format(ff))
    exit
Kf = np.load(ff)[:20,:20]
fj = "{}_K_{}_etkf-jh_cycle{}.npy".format(model, op, 0)
if not os.path.isfile(fj):
    print("not exist {}".format(fj))
    exit
Kj = np.load(fj)[:20,:20]
K = Kf - Kj
ymin = np.min(K)-0.1
ymax = np.max(K)+0.1
ylim = max(np.abs(ymin),ymax)
mappable0 = ax[0,1].pcolor(x,x,K,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
ax[0,1].set_aspect("equal")
ax[0,1].set_xticks(x[::5])
ax[0,1].set_yticks(x[::5])
ax[0,1].set_ylabel("grid point")
ax[0,1].set_xlabel("observation point")
ax[0,1].set_title("K")
pp = fig.colorbar(mappable0, ax=ax[0,1], orientation="vertical")

ff = "{}_K2i_{}_etkf-fh_cycle{}.npy".format(model, op, 0)
if not os.path.isfile(ff):
    print("not exist {}".format(ff))
    exit
Kf = np.load(ff)[:20,:20]
fj = "{}_K2i_{}_etkf-jh_cycle{}.npy".format(model, op, 0)
if not os.path.isfile(fj):
    print("not exist {}".format(fj))
    exit
Kj = np.load(fj)[:20,:20]
K = Kf - Kj
ymin = np.min(K)-0.1
ymax = np.max(K)+0.1
ylim = max(np.abs(ymin),ymax)
mappable0 = ax[1,1].pcolor(x,x,K,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
ax[1,1].set_aspect("equal")
ax[1,1].set_xticks(x[::5])
ax[1,1].set_yticks(x[::5])
ax[1,1].set_ylabel("grid point")
ax[1,1].set_xlabel("observation point")
ax[1,1].set_title("K2 inverse")
pp = fig.colorbar(mappable0, ax=ax[1,1], orientation="vertical")

fig.tight_layout()
fig.savefig("{}_k_{}_etkf.png".format(model,op))
fig.savefig("{}_k_{}_etkf.pdf".format(model,op))
