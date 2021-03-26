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
x = np.arange(na) + 1
y = np.ones(x.shape)
#fig, ax = plt.subplots(2)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,4))
#fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
for pt in perts:
    # trPa
    f = "{}_dpa_{}_{}.txt".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    trpa = np.loadtxt(f)
    # rmse
    f = "{}_e_{}_{}.txt".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    e = np.loadtxt(f)
    ## non-diagonal Pa
    #f = "{}_ndpa_{}_{}.txt".format(model, op, pt)
    #if not os.path.isfile(f):
    #    print("not exist {}".format(f))
    #    continue
    #ndpa = np.loadtxt(f)
    #ax[0].plot(x, trpa, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    ax1.plot(x, trpa, linestyle="solid", color=linecolor[pt], label=pt)
    ax2.plot(x, trpa / e[1:], linestyle="solid", color=linecolor[pt], label=pt)
    #ax1.plot(x, trpa / e[1:], linestyle="solid", color=linecolor[pt], label=pt)
    #ax2.plot(x, e[1:], linestyle="dashed", color=linecolor[pt])
    #r = (ndpa / nx / (nx-1)) / (trpa / nx)
    #ax[1].plot(x, r, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
ax2.plot(x, y, linestyle="dotted", color='tab:purple')
ax1.set_yscale("log")
ax2.set_yscale("log")
#ax1.set_yscale("log")
#ax2.set_yscale("log")
#ax.set_ylim(1e-9,1e-3)
#if op == "quadratic":
#    ax.set_ylim(1e-9,1e-3)
#if op == "cubic":
#    ax.set_ylim(1e-6,1e0)
#ax[0].set_yscale("log")
#ax[1].set_yscale("log")
#if np.max(chi) > 1000.0:
#    ax.set_ylim(0.1, 1000.0)
#    ax.set_yticks([1,10,100,1000])
#if np.max(chi) > 10000.0:
#    ax.set_ylim(0.1, 10000.0)
#    ax.set_yticks([1,10,100,1000,10000])
plt.rcParams['axes.labelsize'] = 16 # fontsize arrange
ax1.set(xlabel="analysis cycle", 
        ylabel="ratio",
        title="trPa {}".format(op))
ax2.set(xlabel="analysis cycle", 
        ylabel="ratio",
        title="trPa / rmse {}".format(op))
ax1.set_xticks(x[::len(x)//10])
ax1.set_xticks(x[::len(x)//20], minor=True)
ax2.set_xticks(x[::len(x)//10])
ax2.set_xticks(x[::len(x)//20], minor=True)
"""
ax1.set(xlabel="analysis cycle", 
        ylabel="ratio",
        title="trPa / rmse {}".format(op))
ax1.set_xticks(x[::len(x)//10])
ax1.set_xticks(x[::len(x)//20], minor=True)
ax2.set(ylabel="rmse")
"""
#ax[0].set(xlabel="analysis cycle", 
#        title="trPa {}".format(op))
#ax[0].set_xticks(x[::len(x)//10])
#ax[0].set_xticks(x[::len(x)//20], minor=True)
#ax[1].set(xlabel="analysis cycle", 
#        title="non-diagPa/diagPa")
#ax[1].set_xticks(x[::len(x)//10])
#ax[1].set_xticks(x[::len(x)//20], minor=True)
#ax[0].legend()
plt.rcParams['legend.fontsize'] = 16 # fontsize arrange
ax1.legend()
ax2.legend()
#ax1.legend()
#fig.tight_layout()
fig.savefig("{}_trpa_{}.png".format(model, op))
fig.savefig("{}_trpa_{}.pdf".format(model, op))
