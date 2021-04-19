import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = sys.argv[4]
if model == "z08" or model == "z05":
    nx = 81
elif model == "l96":
    nx = 40
x = np.arange(nx) + 1
#perts = ["mlef","grad","etkf-jh","etkf-fh"]
cycle = np.arange(na) + 1
ut = np.load("{}_ut.npy".format(model))
#for pt in perts:
    #for j in range(4):
fig, ax = plt.subplots()#2)
f = "{}_ua_{}_{}_cycle{}.npy".format(model, op, pt, 0)
ua = np.load(f)
#ax[0].plot(x,ut[0,:], label="True")
ax.plot(x[:20],ut[0,:20], label="True")
if pt == "mlef" or pt == "grad":
    ax[0].plot(x[:20], ua[:20,0], label="anl cntl")
    for l in range(ua.shape[1]-1):
        ax[1].plot(x[:20], ua[:20,l+1]-ua[:20,0], linestyle="dashed", label="mem{}".format(l+1))
else:
    ua_ = np.mean(ua[:,:], axis=1)
    #ax[0].plot(x[:20], ua_[:20], label="anl mean")
    ax.plot(x[:20], ua_[:20], label="anl mean")
    #for l in range(ua.shape[1]):
    #    ax[1].plot(x[:20], ua[:20,l]-ua_[:20], linestyle="dashed", label="mem{}".format(l+1))
ax.set_xticks(x[:20:5])
ax.set_xticks(x[:20], minor=True)
ax.legend()
#ax[0].set_xticks(x[::5])
#ax[0].set_xticks(x, minor=True)
#ax[1].set_xticks(x[::5])
#ax[1].set_xticks(x, minor=True)
#ax[0].set_title("cycle {}".format(1))
#ax[1].set_title("perturbation")
#ax[0].legend()
#ax[1].legend(loc='upper right')
    #bbox_to_anchor=(1.05,1))    fig.suptitle(op)
#fig.tight_layout(rect=[0,0,1,0.96])
#fig.savefig("{}_du_{}.png".format(model,op))
fig.savefig("{}_ua_{}_{}_first.pdf".format(model,op,pt))
#fig.savefig("{}_ua_{}_{}_cycle{:02d}.pdf".format(model,op,pt,cycle[j]))
plt.close()    
f = "{}_ua_{}_{}.npy".format(model, op, pt)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit()
ua = np.load(f)
f = "{}_uf_{}_{}.npy".format(model, op, pt)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit()
uf = np.load(f)
for j in range(na):
    fig, ax = plt.subplots(2)
    ax[0].plot(x,ut[j,:], label="True")
    if pt == "mlef" or pt == "grad":
        ax[0].plot(x, ua[j,:,0], label="anl cntl")
        ax[0].plot(x, uf[j,:,0], label="fcst cntl")
        for l in range(ua.shape[2]-1):
            ax[1].plot(x, ua[j,:,l+1]-ua[j,:,0], linestyle="dashed", label="mem{}".format(l+1))
    else:
        ua_ = np.mean(ua[j,:,:], axis=1)
        uf_ = np.mean(uf[j,:,:], axis=1)
        ax[0].plot(x, ua_, label="anl mean")
        ax[0].plot(x, uf_, label="fcst mean")
        for l in range(ua.shape[2]):
            ax[1].plot(x, ua[j,:,l]-ua_, linestyle="dashed", label="mem{}".format(l+1))
    ax[0].set_xticks(x[::10])
    ax[0].set_xticks(x[::5], minor=True)
    ax[1].set_xticks(x[::10])
    ax[1].set_xticks(x[::5], minor=True)
    ax[0].set_title("cycle {}".format(cycle[j]))
    ax[1].set_title("perturbation")
    ax[0].legend()
    ax[1].legend(loc='upper right')
            #bbox_to_anchor=(1.05,1))
    fig.suptitle(op)
    fig.tight_layout(rect=[0,0,1,0.96])
    #fig.savefig("{}_du_{}.png".format(model,op))
    fig.savefig("{}_ua_{}_{}_cycle{:02d}.png".format(model,op,pt,cycle[j]))
        #fig.savefig("{}_ua_{}_{}_cycle{:02d}.pdf".format(model,op,pt,cycle[j]))
    plt.close()
