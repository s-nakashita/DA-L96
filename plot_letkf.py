import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
if model == "z08" or model == "z05":
    na = 20
elif model == "l96":
    na = 100
x = np.arange(na) + 1
fig, ax = plt.subplots()
obs = ["all", "half", "quar"]
line = ["-", "--", "dotted"]
i = 0
for ob in obs:
    e = np.loadtxt("{}_e_{}_{}_letkf-{}.txt".format(model, op, "letkf", ob))
    if np.isnan(e).any():
        continue
    ax.plot(x, e, linestyle=line[i], label=ob)
    i += 1
ax.set(xlabel="analysis cycle", ylabel="RMSE",
        title=op)
#ax.set_xticks(x[::5])
#ax.set_xticks(x, minor=True)
ax.legend()
fig.savefig("{}_e_{}_letkf.png".format(model, op))
