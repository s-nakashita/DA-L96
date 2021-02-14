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
#obs_s = float(sys.argv[4])
#oberr = str(int(obs_s*1e4)).zfill(4)
if model == "z08" or model == "z05":
    nx = 81
    perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]
    #na = 20
elif model == "l96":
    nx = 40
    perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]
    #perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
    #na = 300
x = np.arange(nx+1)
lx = np.arange(21) + 1
cmap = "coolwarm"
#plt.rcParams['axes.labelsize'] = 16 # fontsize arrange
fig, ax = plt.subplots(2,2)
pt = perts[0]
#f = "{}_dy_{}_{}_cycle{}_oberr{}.npy".format(model, op, pt, 0, oberr)
f = "{}_dy_{}_{}_cycle{}.npy".format(model, op, pt, 0)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    sys.exit()
dy = np.load(f)
u, sj, vt = la.svd(dy)
print(u.shape)
y = np.arange(u.shape[0])
ax[0,1].plot(y,dy)
ax[0,1].set_xticks(y[::10])
ax[0,1].set_title(pt + " dY")
nmode = sj.size
mode = np.arange(nmode)
for i in range(nmode):
    ax[0,0].plot(y,u[:,i],linestyle='solid')
#ax[0,0].set_xticks(x[::10])
#ax[0,0].set_yticks(y[::10])
#ax[0,0].set_ylabel("observation point")
#ax[0,0].set_xlabel("mode")
#ax[0,0].invert_yaxis()
ax[0,0].set_title(pt+" left singular vectors")
#ax[1,0].bar(mode,s,width=0.35)
#ax[1,0].set_xlabel("mode")
#ax[1,0].set_title(pt+" singular values")

pt = perts[1]
#f = "{}_dy_{}_{}_cycle{}_oberr{}.npy".format(model, op, pt, 0, oberr)
f = "{}_dy_{}_{}_cycle{}.npy".format(model, op, pt, 0)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    sys.exit()
dy = np.load(f)
u, sf, vt = la.svd(dy)
y = np.arange(u.shape[0])
ax[1,1].plot(y,dy)
ax[1,1].set_xticks(y[::10])
ax[1,1].set_title(pt + " dY")
nmode = sf.size 
mode = np.arange(sf.size) + 1
#for i in range(nmode):
#    #ax[0,1].plot(y,u[:,i],linestyle='dashed')
#    ax[0,0].plot(y,u[:,i],linestyle='dashed')
##ax[0,0].set_xticks(x[::10])
##ax[0,1].set_yticks(y[::10])
##ax[0,1].set_ylabel("observation point")
##ax[0,0].set_xlabel("mode")
##ax[0,1].invert_yaxis()
##ax[0,1].set_title(pt+" left singular vectors")
#ax[0,0].set_title("left singular vectors")
width=0.35
ax[1,0].bar(mode-width/2,sj,width=0.35,label=perts[0])
ax[1,0].bar(mode+width/2,sf,width=0.35,label=perts[1])
ax[1,0].set_xlabel("mode")
ax[1,0].set_xticks(mode)
ax[1,0].legend()
ax[1,0].set_title("singular values")
#ax[1,1].bar(mode,s,width=0.35)
#ax[1,1].set_xlabel("mode")
#ax[1,1].set_title(pt+" singular values")
fig.tight_layout()
#fig.savefig("{}_dy_{}_oberr{}.png".format(model,op,oberr))
fig.savefig("{}_dy_{}_mlef.png".format(model,op))

fig, ax = plt.subplots(2,2)
pt = perts[2]
#f = "{}_dy_{}_{}_cycle{}_oberr{}.npy".format(model, op, pt, 0, oberr)
f = "{}_dy_{}_{}_cycle{}.npy".format(model, op, pt, 0)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    sys.exit()
dy = np.load(f)
u, sj, vt = la.svd(dy)
print(u.shape)
y = np.arange(u.shape[0])
ax[0,1].plot(y,dy)
ax[0,1].set_xticks(y[::10])
ax[0,1].set_title(pt + " dY")
nmode = sj.size
mode = np.arange(nmode)
for i in range(nmode):
    ax[0,0].plot(y,u[:,i],linestyle='solid')
#ax[0,0].set_xticks(x[::10])
#ax[0,0].set_yticks(y[::10])
#ax[0,0].set_ylabel("observation point")
#ax[0,0].set_xlabel("mode")
#ax[0,0].invert_yaxis()
ax[0,0].set_title(pt+" left singular vectors")
#ax[1,0].bar(mode,s,width=0.35)
#ax[1,0].set_xlabel("mode")
#ax[1,0].set_title(pt+" singular values")

pt = perts[3]
#f = "{}_dy_{}_{}_cycle{}_oberr{}.npy".format(model, op, pt, 0, oberr)
f = "{}_dy_{}_{}_cycle{}.npy".format(model, op, pt, 0)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    sys.exit()
dy = np.load(f)
u, sf, vt = la.svd(dy)
y = np.arange(u.shape[0])
ax[1,1].plot(y,dy)
ax[1,1].set_xticks(y[::10])
ax[1,1].set_title(pt + " dY")
nmode = sf.size 
mode = np.arange(sf.size) + 1
#for i in range(nmode):
#    #ax[0,1].plot(y,u[:,i])
#    ax[0,0].plot(y,u[:,i],linestyle='dashed')
##ax[0,0].set_xticks(x[::10])
##ax[0,1].set_yticks(y[::10])
##ax[0,1].set_ylabel("observation point")
##ax[0,0].set_xlabel("mode")
##ax[0,1].invert_yaxis()
##ax[0,1].set_title(pt+" left singular vectors")
#ax[0,0].set_title("left singular vectors")
width=0.35
ax[1,0].bar(mode-width/2,sj,width=0.35,label=perts[2])
ax[1,0].bar(mode+width/2,sf,width=0.35,label=perts[3])
ax[1,0].set_xlabel("mode")
ax[1,0].set_xticks(mode)
ax[1,0].legend()
ax[1,0].set_title("singular values")
#ax[1,1].bar(mode,s,width=0.35)
#ax[1,1].set_xlabel("mode")
#ax[1,1].set_title(pt+" singular values")
fig.tight_layout()
#fig.savefig("{}_dy_{}_oberr{}.png".format(model,op,oberr))
fig.savefig("{}_dy_{}_etkf.png".format(model,op))
"""
diff = diff - u
fig, ax = plt.subplots()
ymin = np.min(diff)*1.001
ymax = np.max(diff)*1.001
ylim = max(np.abs(ymin),ymax)
mappable0 = ax.pcolor(x,x,diff,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
ax.set_aspect("equal")
ax.set_xticks(x[::10])
ax.set_yticks(x[::10])
ax.set_ylabel("observation point")
ax.set_xlabel("observation point")
ax.set_title(perts[0]+"-"+perts[1])
pp = fig.colorbar(mappable0, ax=ax, orientation="vertical")
#fig.tight_layout()
fig.savefig("{}_dydiff_{}_oberr{}.png".format(model,op,oberr))
"""