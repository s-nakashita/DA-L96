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
obs_s = 0.001
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
for pt in perts:
    #fig, ax = plt.subplots(2,1)
    fig, ax = plt.subplots(2,2)
    f = "{}_K2i_{}_{}_cycle{}_oberr{}.npy".format(model, op, pt, 0, oberr)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    K = np.load(f)[:20,:20]
    j=0
    ymin = np.min(K)*1.001
    ymax = np.max(K)*1.001
    ylim = max(np.abs(ymin),ymax)
    #mappable0 = ax[j].pcolor(x,x,K,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
    #ax[j].set_aspect("equal")
    #ax[j].set_xticks(x[::5])
    #ax[j].set_yticks(x[::5])
            #ax[j, i].invert_xaxis()
            #ax[j, i].invert_yaxis()
    #ax[j].set_title("Before")
    #pp = fig.colorbar(mappable0, ax=ax[j], orientation="vertical")
    mappable0 = ax[1,1].pcolor(lx,lx,K,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
    ax[1,1].set_aspect("equal")
    ax[1,1].set_xticks(lx[::5])
    ax[1,1].set_yticks(lx[::5])
    ax[1,1].set_ylabel("grid point")
    ax[1,1].set_xlabel("observation point")
    ax[1,1].set_title("K2 inverse")
    pp = fig.colorbar(mappable0, ax=ax[1,1], orientation="vertical")

    f = "{}_K2_{}_{}_cycle{}_oberr{}.npy".format(model, op, pt, 0, oberr)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    K2 = np.load(f)
    w, vr = la.eigh(K2)
    ax[0,0].plot(w)
    #ax[1,0].set_aspect("equal")
    ax[0,0].set_yscale("log")
    ax[0,0].set_xticks(x[::10])
    ax[0,0].set_xlabel("grid point")
    ax[0,0].set_title("eigenvalues")
    j=0
    ymin = np.min(vr)*1.001
    ymax = np.max(vr)*1.001
    ylim = max(np.abs(ymin),ymax)
    #mappable0 = ax[j].pcolor(x,x,K,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
    #ax[j].set_aspect("equal")
    #ax[j].set_xticks(x[::5])
    #ax[j].set_yticks(x[::5])
            #ax[j, i].invert_xaxis()
            #ax[j, i].invert_yaxis()
    #ax[j].set_title("Before")
    #pp = fig.colorbar(mappable0, ax=ax[j], orientation="vertical")
    mappable0 = ax[0,1].pcolor(x,x,vr,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
    ax[0,1].set_aspect("equal")
    ax[0,1].set_xticks(x[::10])
    ax[0,1].set_yticks(x[::10])
    ax[0,1].set_ylabel("grid point")
    ax[0,1].set_xlabel("observation point")
    ax[0,1].set_title("eigenvector")
            #ax[j, i].invert_xaxis()
            #ax[j, i].invert_yaxis()
    pp = fig.colorbar(mappable0, ax=ax[0,1], orientation="vertical")
    wi = 1.0 / w
    K2 = vr @ np.diag(wi) @ vr.T
    ymin = np.min(K2)#-0.1
    ymax = np.max(K2)#+0.1
    ylim = max(np.abs(ymin),ymax)
    #mappable0 = ax[j].pcolor(x,x,K,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
    #ax[j].set_aspect("equal")
    #ax[j].set_xticks(x[::5])
    #ax[j].set_yticks(x[::5])
            #ax[j, i].invert_xaxis()
            #ax[j, i].invert_yaxis()
    #ax[j].set_title("Before")
    #pp = fig.colorbar(mappable0, ax=ax[j], orientation="vertical")
    mappable0 = ax[1,0].pcolor(lx,lx,K2[:20,:20],cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
    ax[1,0].set_aspect("equal")
    ax[1,0].set_xticks(lx[::10])
    ax[1,0].set_yticks(lx[::10])
    ax[1,0].set_ylabel("grid point")
    ax[1,0].set_xlabel("observation point")
    ax[1,0].set_title("K2 inverse (compute)")
            #ax[j, i].invert_xaxis()
            #ax[j, i].invert_yaxis()
    pp = fig.colorbar(mappable0, ax=ax[1,0], orientation="vertical")
    
    #pp.set_clim(ymin,ymax)
    fig.tight_layout()
    fig.savefig("{}_k2_{}_{}_oberr{}.png".format(model,op,pt,oberr))
    #fig.savefig("{}_kb_{}_{}.pdf".format(model,op,pt))

    """
    fig, ax = plt.subplots()
    f = "{}_Kloc_{}_{}_cycle{}.npy".format(model, op, pt, 0)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    Kloc = np.load(f)[:20,:20]
    print(np.min(Kloc),np.max(Kloc))
    j=1
    ymin = np.min(Kloc)-0.1
    ymax = np.max(Kloc)+0.1
    ylim = max(np.abs(ymin),ymax)
    #mappable1 = ax[j].pcolor(x,x,Kloc,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
    #ax[j].set_aspect("equal")
    #ax[j].set_xticks(x[::5])
    #ax[j].set_yticks(x[::5])
            #ax[j, i].invert_xaxis()
            #ax[j, i].invert_yaxis()
    #ax[j].set_title("After")
    #pp = fig.colorbar(mappable1, ax=ax[j], orientation="vertical")
    mappable1 = ax.pcolor(x,x,Kloc,cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
    ax.set_aspect("equal")
    ax.set_xticks(x[::5])
    ax.set_yticks(x[::5])
    ax.set_ylabel("grid point")
    ax.set_xlabel("observation point")
            #ax[j, i].invert_xaxis()
            #ax[j, i].invert_yaxis()
    #axpos = ax[0].get_position()
    #cbar_ax = fig.add_axes([0.90, axpos.y0/4, 0.02, 2*axpos.height])
    #mappable = ScalarMappable(cmap=cmap)
    pp = fig.colorbar(mappable1, ax=ax, orientation="vertical")
    #pp.set_clim(ymin,ymax)
    #fig.tight_layout()
    fig.savefig("{}_kl_{}_{}.png".format(model,op,pt))
    fig.savefig("{}_kl_{}_{}.pdf".format(model,op,pt))
    #fig.savefig("{}_k_{}_{}.png".format(model,op,pt))
    """