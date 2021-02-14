import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
if model == "z08" or model == "z05":
    nx = 81
elif model == "l96":
    nx = 40
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
perts = ["etkf"]
cycle = range(30)
x = np.arange(nx) + 1
for pt in perts:
    fig, ax = plt.subplots()
    dh = np.zeros((len(x), len(cycle)))
    for i in cycle:
        dh_jh = np.load("{}_dh_{}_{}_cycle{}_wIL_JH.npy".format(model, op, pt, i))
        dh_fh = np.load("{}_dh_{}_{}_cycle{}_wIL_fH.npy".format(model, op, pt, i))
        dh[:,i] = dh_jh[:,0] - dh_fh[:,0]
    t = np.array(cycle) + 1
    ax.pcolor(x, t, dh.T, cmap="RdBu_r")
    ax.set(xlabel="site", ylabel="cycle")
    ax.set_yticks(t[::5])
    fig.suptitle("{}, JH-fullH".format(pt))
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig("{}_dhdiff_{}_{}.png".format(model, op, pt))
