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
elif model == "z08":
    perts = ["mlef", "grad", "mlefb", "mleft", "etkf", "etkf-fh", "etkf-jh"]#, "po", "srf", "letkf"]
    linestyle = {"mlef":"solid", "grad":"dashed",
     "mlefb":"solid", "mleft":"solid", "etkf":"solid",
     "etkf-fh":"solid", "etkf-jh":"dashed"}
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',
    "etkf":'tab:green', "mlefb":"tab:cyan", "mleft":"tab:pink",
    "etkf-fh":'tab:green',"etkf-jh":'tab:red'}
x = np.arange(na) + 1
y = np.ones(x.shape)
fig, ax = plt.subplots()
for pt in perts:
    f = "{}_chi_{}_{}.txt".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    chi = np.loadtxt(f)
    ax.plot(x, chi, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
ax.plot(x, y, linestyle="dotted", color='tab:purple')
ax.set_yscale("log")
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
