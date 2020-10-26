import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["etkf", "po", "srf", "letkf"]
perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
if model == "l96" and op == "linear":
    perts = ["mlef", "etkf", "po", "srf", "letkf"]
    #na = 100
x = np.arange(na) + 1
y = np.ones(x.shape)
fig, ax = plt.subplots()
for pt in perts:
    f = "{}_chi_{}_{}.txt".format(model, op, pt)
    if not os.path.isfile(f):
        continue
    chi = np.loadtxt(f)
    ax.plot(x, chi, label=pt)
ax.plot(x, y, linestyle="dotted")
#ax.set_yscale("log")
#if np.max(chi) > 1000.0:
#    ax.set_ylim(0.1, 1000.0)
#    ax.set_yticks([1,10,100,1000])
#if np.max(chi) > 10000.0:
#    ax.set_ylim(0.1, 10000.0)
#    ax.set_yticks([1,10,100,1000,10000])
ax.set(xlabel="analysis cycle", ylabel="Chi2",
        title=op)
ax.set_xticks(x[::len(x)//10])
ax.set_xticks(x[::len(x)//20], minor=True)
ax.legend()
fig.savefig("{}_chi_{}.png".format(model, op))
