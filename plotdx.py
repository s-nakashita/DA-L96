import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
if model == "z08" or model == "z05":
    nx = 81
    perts = ["mlef", "grad", "etkf-jh", "etkf-fh"]
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf-fh":'tab:green',"etkf-jh":'tab:red'}
elif model == "l96":
    nx = 40
    perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
#perts = ["etkf"]
cycle = range(4)
x = np.arange(nx) + 1
fig, ax = plt.subplots(2,2)
for j in range(2):
    for i in range(2):
        icycle = 2*j+i
        for pt in perts:
            f = "{}_dx_{}_{}_cycle{}.npy".format(model, op, pt, cycle[icycle])
            if not os.path.isfile(f):
                print("not exist {}".format(f))
                continue
            dx = np.load(f)
            ax[j, i].plot(x, dx, color=linecolor[pt], label=pt)
        ax[j, i].set_xticks(x[::10])
        ax[j, i].set_xticks(x[::5], minor=True)
        ax[j, i].set_title("cycle{}".format(cycle[icycle]))
if op == "quadratic" :
    ax[0, 0].set_ylim(-2.0,2.0)
elif op == "cubic":
    ax[0, 0].set_ylim(-3.0,3.0)
elif op == "quadratic-nodiff":
    ax[0, 0].set_ylim(-0.6,0.6)
elif op == "cubic-nodiff":        
    ax[0, 0].set_ylim(-0.75,0.75)        
ax[0, 1].legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
fig.suptitle(op)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("{}_dx_{}.png".format(model, op))
