import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
if model == "z08" or model == "z05":
    nx = 81
    perts = ["etkf-jh", "etkf-fh_av-op", "etkf-fh_op-av"]
    sigma = {"linear": 1.0e-3, "quadratic": 1.0e-3, "cubic": 1.0e-3, "quartic": 1.0e-3, \
    "quadratic-nodiff": 1.0e-3, "cubic-nodiff": 1.0e-3, "quartic-nodiff": 1.0e-3, "test":8.0e-2}
elif model == "l96":
    nx = 40
    perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
#perts = ["etkf"]
cycle = range(na)
x = np.arange(nx) + 1
cmap = "coolwarm"
rmat = np.eye(nx) / sigma[op]
rinv = rmat.transpose() @ rmat
for pt in perts:
    print(pt)
    #for j in range(2):
    for i in range(1):
        plt.rcParams['axes.labelsize'] = 16 # fontsize arrange
        fig, ax = plt.subplots(1,2, figsize=(9, 4))
        icycle = i
        f = "{}_dxf_{}_{}_cycle{}.npy".format(model, op, pt, cycle[icycle])
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        dxf = np.load(f)
        print(dxf.T @ dxf)
        for k in range(dxf.shape[1]):
            ax[0].plot(x[:20], dxf[:20, k], label="mem{}".format(k+1))
        ax[0].set_xticks(x[:20:5])
        #ax[i, 0].set_xticks(x[::5], minor=True)
        ax[0].set_title("dXf cycle{}".format(cycle[icycle]))
        f = "{}_dh_{}_{}_cycle{}.npy".format(model, op, pt, cycle[icycle])
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        dh = np.load(f)
        for k in range(dh.shape[1]):
            ax[1].plot(x[:20], dh[:20, k], label="mem{}".format(k))
        ax[1].set_xticks(x[:20:5])
        #ax[i, 1].set_xticks(x[::5], minor=True)
        ax[1].set_title("dY cycle{}".format(cycle[icycle]))
        ax[0].legend()#bbox_to_anchor=(1.05, 1.05), loc='upper right')
        #ax[1].legend()#bbox_to_anchor=(1.05, 1.05), loc='upper left')
        fig.suptitle(op)
        fig.tight_layout()#rect=[0,0,1,0.96])
#        fig.savefig("{}_dh_{}_{}_cycle{}.png".format(model, op, pt, cycle[icycle]))
        plt.close()

        fig = plt.figure(figsize=(9,9))
        gs = GridSpec(2,2, figure=fig)
        fig2, ax = plt.subplots()
        nmem = dxf.shape[1]
        dxf = dxf / np.sqrt(nmem-1)
        dh = dh / np.sqrt(nmem-1)
        zmat = rmat @ dh
        #cmat = zmat.transpose() @ zmat + (zmat.shape[1]-1)*np.eye(zmat.shape[1])
        #lam, v = la.eigh(cmat)
        #print("{} {} eigenvalues {}".format(op, pt, lam))
        #cinv = v[:,1:] @ np.diag(1.0/lam[1:]) @ v[:,1:].transpose()
        #dx = dxf @ v[:,1:]
        #dy = dh @ v[:,1:]
        #K = dx @ np.diag(1.0/lam[1:]) @ dy.transpose() @ rinv
        u, s, vt = la.svd(zmat)
        print(s)
        v = vt.transpose()
        print(np.dot(v[:,-1], v[:,-2]))
        lam = s / (1+s**2) / sigma[op]
        dx = dxf @ v
        dy = u[:,:zmat.shape[1]] 
        print(dx.shape)
        print(dy.shape)
        K = dx @ np.diag(lam) @ dy.transpose()
        #dof = np.sum(s>1e-10)
        #dh_inv = vt.transpose()[:,:dof] @ np.diag(1.0/s[:dof]) @ u.transpose()[:dof,:]
        #print(dof)
        #print(np.diag(dh@dh_inv))
        #print(dh_inv@dh)
        #K = dxf @ dh_inv
        ymin = np.min(K)-0.1
        ymax = np.max(K)+0.1
        ylim = max(np.abs(ymin),ymax)
        ax0 = fig.add_subplot(gs[0,0])
        mappable0 = ax0.pcolor(x[:21],x[:21],K[:20,:20],cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
        ax0.set_aspect("equal")
        ax0.set_xticks(x[:21:5])
        ax0.set_yticks(x[:21:5])
        ax0.set_ylabel("grid point")
        ax0.set_xlabel("observation point")
        ax0.set_title("K")
        pp = fig.colorbar(mappable0, ax=ax0, orientation="vertical")
        ax1 = fig.add_subplot(gs[0,1])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])
        ax1.bar(np.arange(1, len(lam)+1)-0.15, lam, width=0.3, color="tab:blue", label="scale")
        ax.bar(np.arange(1, len(lam)+1)-0.15, lam, width=0.3, color="tab:red", label="scale")
        ax12 = ax1.twinx()
        ax_ = ax.twinx()
        ax12.bar(np.arange(1, len(s)+1)+0.15, s, width=0.3, color="tab:red", label="singular value")
        ax_.bar(np.arange(1, len(s)+1)+0.15, s, width=0.3, color="tab:blue", label="singular value")
        for k in range(dx.shape[1]):
            #print("mode{} max{} min{}".format(k, np.max(dh_inv[k]), np.min(dh_inv[k])))
            #ax[1].plot(x[:20], dh_inv[k, :20], label="mode{}".format(k))
            #ax2.plot(np.arange(1, v.shape[1]+1), v[:,k], label="mode{}".format(k+1))
            ax2.plot(x[:21], dx[:21,k], label="mode{}".format(k+1))
            ax3.plot(x[:21], dy[:21,k], label="mode{}".format(k+1))
        ax1.set_title("scale")
        ax.set_title("singular value")
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        ax12.tick_params(axis='y', labelcolor="tab:red")
        ax_.tick_params(axis='y', labelcolor="tab:blue")
        ax.tick_params(axis='y', labelcolor="tab:red")
        if np.log10(np.max(lam)) > 3.0:
            ax1.set_yscale("log")
            ax.set_yscale("log")
        if np.log10(np.max(s)) > 3.0:
            ax12.set_yscale("log")
            ax_.set_yscale("log")
            smax = np.power(10, int(np.log10(np.max(s))+1))
            ax12.set_ylim(0.01, smax)
            ax_.set_ylim(0.01, smax)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax12.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper center')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc='upper center')
        ax2.set_xticks(x[:21:5])
        #ax2.set_xticks(np.arange(1, v.shape[1]+1))
        ax3.set_xticks(x[:21:5])
        ax2.set_title("dXf @ V")
        #ax2.set_title("V")
        #ax3.set_title("dY @ V")
        ax3.set_title("U")
        ax2.legend()
        ax3.legend()
        #ax[1].set_xticks(x[:20:5])
        #ax[i, 1].set_xticks(x[::5], minor=True)
        #ax[1].set_title("inverse dY cycle{}".format(cycle[icycle]))
        #ax[1].legend()#bbox_to_anchor=(1.05, 1.05), loc='upper right')
        #ax[1].legend()#bbox_to_anchor=(1.05, 1.05), loc='upper left')
        #fig.suptitle(op)
        fig.tight_layout()#rect=[0,0,1,0.96])
        fig.savefig("{}_kuv_{}_{}_cycle{}.png".format(model,op,pt,i))
        fig.savefig("{}_kuv_{}_{}_cycle{}.pdf".format(model,op,pt,i))
        #fig2.savefig("{}_sg_{}_{}_cycle{}.pdf".format(model,op,pt,i))
        plt.close() 

        plt.rcParams['axes.labelsize'] = 12 # fontsize arrange
        fig = plt.figure(figsize=(9,6))
        gs = GridSpec(2,3, figure=fig)
        ii = 0
        jj = 0
        xaxis = np.arange(nx+1)+1
        for k in range(dx.shape[1]):
            Kp = dx[:,k].reshape(-1,1) @ dy[:,k].reshape(1,-1) * lam[k]
            ymin = np.min(Kp)
            ymax = np.max(Kp)
            ylim = max(np.abs(ymin),ymax)
            ax = fig.add_subplot(gs[ii,jj])
            mappable0 = ax.pcolor(xaxis[:21],xaxis[:21],Kp[:20,:20],cmap=cmap,norm=Normalize(vmin=-ylim,vmax=ylim))
            ax.set_aspect("equal")
            ax.set_xticks(xaxis[:21:5])
            ax.set_yticks(xaxis[:21:5])
            ax.set_ylabel("grid point")
            ax.set_xlabel("observation point")
            ax.set_title("mode{}".format(k+1))
            pp = fig.colorbar(mappable0, ax=ax, orientation="vertical")
            jj += 1
            if jj >= 3:
                ii += 1
                jj = 0
        fig.tight_layout()#rect=[0,0,1,0.96])
        fig.savefig("{}_kpert_{}_{}_cycle{}.png".format(model,op,pt,i))
        fig.savefig("{}_kpert_{}_{}_cycle{}.pdf".format(model,op,pt,i))
        plt.close() 