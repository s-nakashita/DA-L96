import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["etkf", "po", "srf", "letkf"]
nx = 40
perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
if model == "l96" and op == "linear":
    perts = ["mlef", "etkf", "po", "srf", "letkf"]
    #na = 100
elif model == "z08":
    nx = 81
    perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]#, "po", "srf", "letkf"]
    linestyle = {"mlef":"solid", "grad":"dashed",
     "etkf-fh":"solid", "etkf-jh":"dashed"}
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf-fh":'tab:green',"etkf-jh":'tab:red'}
x = np.arange(na+1)
#y = np.ones(x.shape)
#fig, ax = plt.subplots(2)
fig, ax = plt.subplots()
for pt in perts:
    # trPa
    f = "{}_dpf_{}_{}.txt".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    trpf = np.loadtxt(f)
    ax.plot(x, trpf, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    
ax.set_yscale("log")
#ax.set_ylim(1e-9,1e-3)
plt.rcParams['axes.labelsize'] = 16 # fontsize arrange
ax.set(xlabel="analysis cycle", 
        title="trPa {}".format(op))
ax.set_xticks(x[::len(x)//10])
ax.set_xticks(x[::len(x)//20], minor=True)
plt.rcParams['legend.fontsize'] = 16 # fontsize arrange
ax.legend()
#fig.tight_layout()
fig.savefig("{}_trpf_{}.png".format(model, op))
#fig.savefig("{}_trpf_{}.pdf".format(model, op))
