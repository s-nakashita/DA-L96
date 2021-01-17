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
perts = ["mlef","grad","etkf-fh","etkf-jh"]
linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf-fh":'tab:green',"etkf-jh":'tab:red'}
cycle = range(4)#[30, 60, 90, 120]
fig, ax = plt.subplots(2,2)
for j in range(2):
    for i in range(2):
        k = 2*j+i
        ut = np.load("{}_ut.npy".format(model))[k,:]
        for pt in perts:
            f = "{}_ua_{}_{}_cycle{}.npy".format(model, op, pt, k)
            if not os.path.isfile(f):
                continue
            du_jh = np.load(f)[:,0] - ut
            #du_fh = np.load("{}_ua_{}_{}_{}_wL.npy".format(model, op, pt))[cycle[k],:,0] - ut
            print(du_jh.argmax())
            ax[j, i].plot(x, du_jh, color=linecolor[pt], label=pt)
            #ax[j, i].plot(x, du_fh, label="full H")
        ax[j, i].set_xticks(x[::10])
        ax[j, i].set_xticks(x[::5], minor=True)
ax[0, 0].legend()
fig.suptitle(op)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("{}_du_{}.png".format(model,op))
#fig.savefig("{}_du_{}_etkf.png".format(model,op))
