import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
if model == "z08" or model == "z05":
    #perts = ["mlef", "mlefb"]
    perts = ["mlef", "grad", "mlefb", "gradb", "etkf-fh", "etkf-jh"]#, "po", "srf", "letkf"]
    linestyle = {"mlef":"solid", "grad":"dashed",
     "mlefb":"solid", "gradb":"dashed",
     "etkf-fh":"solid", "etkf-jh":"dashed"}
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"mlefb":'tab:cyan',"gradb":'tab:pink',"etkf-fh":'tab:green',"etkf-jh":'tab:red'}
    #na = 20
    #sigma = {"linear": 8.0e-2, "quadratic": 8.0e-2, "cubic": 7.0e-4, "quartic": 7.0e-4,\
    #"quadratic-nodiff": 8.0e-2, "cubic-nodiff": 7.0e-4, "quartic-nodiff": 7.0e-4}
    sigma = {"linear": 8.0e-2, "quadratic": 1.0e-3, "cubic": 1.0e-3, "quartic": 1.0e-2, \
    "quadratic-nodiff": 1.0e-3, "cubic-nodiff": 1.0e-3, "quartic-nodiff": 1.0e-2}
    x = np.arange(na+1)
    #x = np.arange(na) + 1
elif model == "l96":
    if op == "linear":
        perts = ["mlef", "etkf", "po", "srf", "letkf"]
    else:
        perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
    #perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
    linestyle = {"mlef":"solid", "grad":"solid", "etkf":"solid", "po":'solid',\
        "srf":"solid", "letkf":"solid"}
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf":'tab:green', "po":'tab:red',\
        "srf":"tab:pink", "letkf":"tab:purple", "kf":"tab:cyan", "var":"tab:olive",\
        "var4d":"tab:brown"    }
    #perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]
    #linestyle = {"mlef":"solid", "grad":"dashed",
    # "etkf-fh":"solid", "etkf-jh":"dashed"}
    #linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf-fh":'tab:green',"etkf-jh":'tab:red'}
    sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0}
    #sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    #"quadratic-nodiff": 1.0, "cubic-nodiff": 1.0, "test":1.0}
    x = np.arange(na) + 1
y = np.ones(x.size) * sigma[op]
fig, ax = plt.subplots()
i = 0
for pt in perts:
    f = "{}_e_{}_{}.txt".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    e = np.loadtxt(f)
    if np.isnan(e).any():
        continue
    ax.plot(x, e, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    #ax.plot(x, e, linestyle="solid", color=linecolor[pt], label=pt)
    #f = "{}_e_{}-nodiff_{}.txt".format(model, op, pt)
    #if not os.path.isfile(f):
    #    print("not exist {}".format(f))
    #    continue
    #e = np.loadtxt(f)
    #if np.isnan(e).any():
    #    continue
    #ax.plot(x, e, linestyle="dashed", color=linecolor[pt], label="{}-nodiff".format(pt))
    i += 1
ax.plot(x, y, linestyle="dotted", color='tab:purple')
plt.rcParams['axes.labelsize'] = 16 # fontsize arrange
ax.set(xlabel="analysis cycle", ylabel="RMSE",
        title=op)
if model == "z08":
    ax.set_ylim(-0.01,0.2)
ax.set_xticks(x[::len(x)//10])
ax.set_xticks(x[::len(x)//20], minor=True)
plt.rcParams['legend.fontsize'] = 16 # fontsize arrange
ax.legend()
fig.savefig("{}_e_{}.pdf".format(model, op))
fig.savefig("{}_e_{}.png".format(model, op))
#fig.savefig("{}_e_{}+nodiff.png".format(model, op))
#fig.savefig("{}_e_{}+nodiff.pdf".format(model, op))
