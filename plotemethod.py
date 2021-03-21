import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#pt = sys.argv[4]
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
perts = ["mlef","grad"]
#perts = ["etkf-fh","etkf-jh"]
if model == "z08":
    sigma = {"linear": 1.0e-3, "quadratic": 1.0e-3, "cubic": 1.0e-3, "quartic": 1.0e-3, \
    "quadratic-nodiff": 1.0e-3, "cubic-nodiff": 1.0e-3, "quartic-nodiff": 1.0e-3, "test":8.0e-2}
    x = np.arange(na+1)
elif model == "l96":
    sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0, "abs":1.0}
    x = np.arange(na)+1
fig, ax = plt.subplots()
methods = ["lb","cgf_fr","cgf_pr","cgf_prb"]
linecolor={"lb":"tab:blue","cgf_fr":"tab:orange","cgf_pr":"tab:green","cgf_prb":"tab:red"}
linestyle=["solid","dashed"]
y = np.ones(x.size) * sigma[op]
j = 0
for method in methods:
    i = 0
    for pt in perts:
        f = "{}_e_{}_{}_{}.txt".format(model, op, pt, method)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        e = np.loadtxt(f)
        ax.plot(x[1:], e[1:], linestyle=linestyle[i], color=linecolor[method], label=pt+"-"+method)
        i += 1
        j += 1
        print(f"pt={pt}, method={method}, RMSE={np.mean(e[1:])}")
ax.plot(x[1:], y[1:], linestyle="dotted", color="tab:purple")
plt.rcParams['axes.labelsize'] = 16 # fontsize arrange
ax.set(xlabel="analysis cycle", ylabel="RMSE",
        title=op)
ax.set_xticks(x[1::5])
ax.set_xticks(x, minor=True)
#if model == "z08":
#    ax.set_ylim(-0.01,0.2)
plt.rcParams['legend.fontsize'] = 16 # fontsize arrange
ax.legend(ncol=2)
fig.savefig("{}_emethod_{}.png".format(model, op))
ax.set_yscale("log")
ax.set_ylim(1e-5,0.2)
fig.savefig("{}_emethod_log_{}.png".format(model, op))