import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
if model == "z08" or model == "z05":
    nx = 81
    perts = ["mlef", "grad", "etkf-jh", "etkf-fh"]
elif model == "l96":
    nx = 40
    perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
#perts = ["etkf"]
cycle = range(4)
x = np.arange(nx) + 1
for pt in perts:
    fig, ax = plt.subplots(2,2)
    for j in range(2):
        for i in range(2):
            icycle = 2*j+i
            f = "{}_d_{}_{}_cycle{}.npy".format(model, op, pt, cycle[icycle])
            if not os.path.isfile(f):
                print("not exist {}".format(f))
                continue
            d = np.load(f)
            ax[j, i].plot(x[:20], d[:20])
            #ax[j, i].set_xticks(x[::5])
            #ax[j, i].set_xticks(x[::5], minor=True)
            ax[j, i].set_title("cycle{}".format(cycle[icycle]))
    #ax[0, 1].legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
    fig.suptitle(op)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig("{}_d_{}_{}.png".format(model, op, pt))
