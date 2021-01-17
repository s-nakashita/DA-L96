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
x = np.arange(20) + 1
perts = ["mlef","grad","etkf-jh","etkf-fh"]
cycle = range(4)
ut = np.load("{}_ut.npy".format(model))
for pt in perts:
    fig, ax = plt.subplots(2,2)
    for j in range(2):
        f = "{}_ua_{}_{}_cycle{}.npy".format(model, op, pt, cycle[j])
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        ua = np.load(f)
        ax[j, 0].plot(x,ut[cycle[j],:20], label="True")
        if pt == "mlef" or pt == "grad":
            ax[j, 0].plot(x, ua[:20,0], label="cntl")
        else:
            ax[j, 0].plot(x, ua[:20,0], label="mean")
        for l in range(4):
            ax[j, 1].plot(x, ua[:20,l+1]-ua[:20,0], linestyle="dashed", label="mem{}".format(l+1))
        ax[j, 0].set_xticks(x[::5])
        ax[j, 0].set_xticks(x, minor=True)
        ax[j, 1].set_xticks(x[::5])
        ax[j, 1].set_xticks(x, minor=True)
        ax[j, 0].set_title("cycle {}".format(j+1))
        ax[j, 1].set_title("perturbation")
    ax[0, 0].legend()
    ax[0, 1].legend(bbox_to_anchor=(1.05,1))
    fig.suptitle(op)
    fig.tight_layout(rect=[0,0,1,0.96])
    #fig.savefig("{}_du_{}.png".format(model,op))
    fig.savefig("{}_ua_{}_{}.png".format(model,op,pt))
    fig.savefig("{}_ua_{}_{}.pdf".format(model,op,pt))
