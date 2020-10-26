import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
if model == "z08" or model == "z05":
    #perts = ["mlef", "grad"]
    perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
    #na = 20
    sigma = {"linear": 8.0e-2, "quadratic": 8.0e-2, "cubic": 7.0e-4, \
    "quadratic-nodiff": 8.0e-2, "cubic-nodiff": 7.0e-4,}
elif model == "l96":
    if op == "linear":
        perts = ["mlef", "etkf", "po", "srf", "letkf"]
    else:
        #perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
        perts = ["mlef", "grad"]
    #na = 100
    sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0}
x = np.arange(na) + 1
y = np.ones(x.size) * sigma[op]
fig, ax = plt.subplots()
for pt in perts:
    f = "{}_e_{}_{}.txt".format(model, op, pt)
    if not os.path.isfile(f):
        continue
    e = np.loadtxt(f)
    if np.isnan(e).any():
        continue
    ax.plot(x, e, label=pt)
ax.plot(x, y, linestyle="dotted")
ax.set(xlabel="analysis cycle", ylabel="RMSE",
        title=op)
ax.set_xticks(x[::len(x)//10])
ax.set_xticks(x[::len(x)//20], minor=True)
ax.legend()
fig.savefig("{}_e_{}.png".format(model, op))
