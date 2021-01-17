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
#sigma=[0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 
#  0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
#for obs_s in sigma:
oberr = str(int(obs_s*1e4)).zfill(4)
r = np.eye(nx) * obs_s * obs_s
rinv = np.eye(nx) / obs_s / obs_s
fig, ax = plt.subplots(2,2)
pt = "etkf-jh"
f = "{}_K2_{}_{}_cycle{}_oberr{}.npy".format(model, op, pt, 0, oberr)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
K2 = np.load(f)
b = -K2 + r
u, sj, vt = la.svd(b)
tinv = vt @ rinv @ u - np.diag(1.0/sj)
tj = la.inv(tinv)
ax[0,0].plot(sj[:10])
ax[0,0].set_xticks(x[:10])
ax[0,0].set_xlabel("observation point")
ax[0,0].set_title(pt+" s")
#ymin = np.min(tj)*1.001
#ymax = np.max(tj)*1.001
#ylim = max(np.abs(ymin),ymax)
#mappable0 = ax[0,1].pcolor(x[:11],x[:11],tj[:10,:10],cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
ax[0,1].plot(np.diag(tj)[:10])
#ax[0,1].set_aspect("equal")
ax[0,1].set_xticks(x[:10])
#ax[0,1].set_yticks(x[:11])
#ax[0,1].set_ylabel("observation point")
ax[0,1].set_xlabel("observation point")
ax[0,1].set_title(pt+" t")
#pp = fig.colorbar(mappable0, ax=ax[0,1], orientation="vertical")
dif = u @ tj @ vt
pt = "etkf-fh"
f = "{}_K2_{}_{}_cycle{}_oberr{}.npy".format(model, op, pt, 0, oberr)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
K2 = np.load(f)
b = -K2 + r
u, sf, vt = la.svd(b)
tinv = vt @ rinv @ u - np.diag(1.0/sf)
tf = la.inv(tinv)
ax[1,0].plot(sf[:10])
ax[1,0].set_xticks(x[:10])
ax[1,0].set_xlabel("observation point")
ax[1,0].set_title(pt+" s")
#ymin = np.min(tf)*1.001
#ymax = np.max(tf)*1.001
#ylim = max(np.abs(ymin),ymax)
#mappable0 = ax[1,1].pcolor(x[:11],x[:11],tf[:10,:10],cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
ax[1,1].plot(np.diag(tf)[:10])
#ax[1,1].set_aspect("equal")
ax[1,1].set_xticks(x[:10])
#ax[1,1].set_yticks(x[:11])
#ax[1,1].set_ylabel("observation point")
ax[1,1].set_xlabel("observation point")
ax[1,1].set_title(pt+" t")
#pp = fig.colorbar(mappable0, ax=ax[1,1], orientation="vertical")
fig.tight_layout()
fig.savefig("{}_smw_{}_oberr{}.png".format(model,op,oberr))

dif = dif - u @ tf @ vt
fig, ax = plt.subplots()
ymin = np.min(dif)*1.001
ymax = np.max(dif)*1.001
ylim = max(np.abs(ymin),ymax)
mappable0 = ax.pcolor(lx,lx,dif[:20,:20],cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
ax.set_aspect("equal")
ax.set_xticks(lx)
ax.set_yticks(lx)
ax.set_ylabel("observation point")
ax.set_xlabel("observation point")
ax.set_title("difference FH-JH")
pp = fig.colorbar(mappable0, ax=ax, orientation="vertical")
fig.tight_layout()
fig.savefig("{}_smwdif_{}_oberr{}.png".format(model,op,oberr))
