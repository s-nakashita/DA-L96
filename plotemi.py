import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
perts = ["mlef", "grad"]
#perts = ["etkf-fh","etkf-jh"]
#if model == "z08":
#    na = 20
#elif model == "l96":
#    na = 100
fig, ax = plt.subplots()
maxiter = [1, 5, 10, 15, 20]

linecolor={"mlef":"tab:blue","grad":"tab:orange","etkf-fh":"tab:green","etkf-jh":"tab:red"}
linestyle=["solid","dashed"]
j = 0
for pt in perts:
    i = 0
    el = np.zeros(len(maxiter))
    eld = np.zeros(len(maxiter))
    for mi in maxiter:
        f = "{}_e_{}_{}_mi{}.txt".format(model, op, pt, mi)
        #f = "{}_e_{}_{}_oberr{}.txt".format(model, op, pt, oberr)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            el[i] = np.nan
            i += 1
            continue
        e = np.loadtxt(f)
        el[i] = np.mean(e[1:])
        i += 1
    ax.plot(maxiter, el, linestyle=linestyle[0], color=linecolor[pt], label=pt)
    i = 0
    for mi in maxiter:
        e = np.loadtxt("{}_e_{}-nodiff_{}_mi{}.txt".format(model, op, pt, mi))
        eld[i] = np.mean(e[5:])
        i += 1
    ax.plot(maxiter, eld, linestyle=linestyle[1], color=linecolor[pt], label="{}-nodiff".format(pt))
    j += 1
ax.set(xlabel="maximum iteration number", ylabel="RMSE",
        title=op)
#ax.set_ylim(0.0, 0.1)
#ax.set_xscale("log")
#ax.set_yscale("log")
#ax.set_xticks(x[::5])
#ax.set_xticks(x, minor=True)
ax.legend()
fig.savefig("{}_emi_{}+nodiff.png".format(model, op))
fig.savefig("{}_emi_{}+nodiff.pdf".format(model, op))
