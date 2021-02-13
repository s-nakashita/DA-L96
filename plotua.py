import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
if model == "z08" or model == "z05":
    nx = 81
elif model == "l96":
    nx = 40
x = np.arange(nx) + 1
perts = ["mlef","grad","mlefb","etkf","etkf-jh","etkf-fh"]
cycle = np.arange(na) + 1
ut = np.load("{}_ut.npy".format(model))
for pt in perts:
    for j in range(na):
        fig, ax = plt.subplots(2)
        f = "{}_ua_{}_{}_cycle{}.npy".format(model, op, pt, j)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        ua = np.load(f)
        ax[0].plot(x,ut[j,:], label="True")
        if pt == "mlef" or pt == "grad":
            ax[0].plot(x, ua[:,0], label="cntl")
            for l in range(4):
                ax[1].plot(x, ua[:,l+1]-ua[:,0], linestyle="dashed", label="mem{}".format(l+1))
        else:
            ua_ = np.mean(ua, axis=1)
            ax[0].plot(x, ua_, label="mean")
            for l in range(4):
                ax[1].plot(x, ua[:,l+1]-ua_, linestyle="dashed", label="mem{}".format(l+1))
        ax[0].set_xticks(x[::10])
        ax[0].set_xticks(x[::5], minor=True)
        ax[1].set_xticks(x[::10])
        ax[1].set_xticks(x[::5], minor=True)
        ax[0].set_title("cycle {}".format(cycle[j]))
        ax[1].set_title("perturbation")
        ax[0].legend()
        ax[1].legend(bbox_to_anchor=(1.05,1))
        fig.suptitle(op)
        fig.tight_layout(rect=[0,0,1,0.96])
    #fig.savefig("{}_du_{}.png".format(model,op))
        fig.savefig("{}_ua_{}_{}_cycle{:02d}.png".format(model,op,pt,cycle[j]))
        #fig.savefig("{}_ua_{}_{}_cycle{:02d}.pdf".format(model,op,pt,cycle[j]))
        plt.close()
