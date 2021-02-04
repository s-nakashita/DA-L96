import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
perts = ["mlef", "etkf"]
#perts = ["etkf-fh","etkf-jh"]
#if model == "z08":
#    na = 20
#elif model == "l96":
#    na = 100
fig, ax = plt.subplots()
#lags = [4, 6, 8, 10, 12]
gamma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
linecolor={"mlef":"tab:blue","etkf":"tab:orange"}
linestyle=["solid","dashed"]
j = 0
for pt in perts:
    i = 0
    el = np.zeros(len(gamma))
    eld = np.zeros(len(gamma))
    for ga in gamma:
        f = "{}_e_{}_{}_ga{}_mean.txt".format(model, op, pt, ga)
        #f = "{}_e_{}_{}_oberr{}.txt".format(model, op, pt, oberr)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            el[i] = np.nan
            i += 1
            continue
        e = np.loadtxt(f)
        #el[i] = np.mean(e[int(na/3):])
        el[i] = np.mean(e[200:])
        i += 1
    ax.plot(gamma, el, linestyle=linestyle[0], color=linecolor[pt], label=pt)
    j += 1
#ax.plot(obs_s, obs_s, linestyle="dotted", color="tab:purple")
ax.set(xlabel="nonlinear observation parameter gamma", ylabel="RMSE",
        title=op)
#ax.set_ylim(0.0, 0.1)
#ax.set_xscale("log")
#ax.set_yscale("log")
#ax.set_xticks(x[::5])
#ax.set_xticks(x, minor=True)
ax.legend()
fig.savefig("{}_egamma_{}.png".format(model, op))
#fig.savefig("{}_eoberr_{}+nodiff.pdf".format(model, op))
