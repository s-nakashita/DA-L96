import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = sys.argv[3]
if model == "z08" or model == "z05":
    nx = 81
elif model == "l96":
    nx = 40
x = np.arange(nx) + 1
perts = ["etkf"]
cycle = [30, 60, 90, 120]
for pt in perts:
    fig, ax = plt.subplots(2,2)
    for j in range(2):
        for i in range(2):
            k = 2*j+i
            ut = np.load("{}_ut.npy".format(model))[cycle[k],:]
            du_jh = np.load("{}_ua_{}_{}_wIL_JH.npy".format(model, op, pt))[cycle[k],:,0] - ut
            du_fh = np.load("{}_ua_{}_{}_wIL_fH.npy".format(model, op, pt))[cycle[k],:,0] - ut
        
            ax[j, i].plot(x, du_jh, label="JH")
            ax[j, i].plot(x, du_fh, label="full H")
            ax[j, i].set_xticks(x[::10])
            ax[j, i].set_xticks(x[::5], minor=True)
    ax[0, 0].legend()
    fig.suptitle(op)
    fig.tight_layout(rect=[0,0,1,0.96])
    #fig.savefig("{}_du_{}.png".format(model,op))
    fig.savefig("{}_du_{}_etkf.png".format(model,op))
