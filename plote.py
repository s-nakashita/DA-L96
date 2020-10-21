import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
#perts = ["mlef", "grad"]
if model == "z08":
    na = 20
elif model == "l96":
    na = 100
x = np.arange(na) + 1
fig, ax = plt.subplots()
for pt in perts:
    e = np.loadtxt("{}_e_{}_{}.txt".format(model, op, pt))
    ax.plot(x, e, label=pt)
ax.set(xlabel="analysis cycle", ylabel="RMSE",
        title=op)
#ax.set_xticks(x[::5])
#ax.set_xticks(x, minor=True)
ax.legend()
fig.savefig("{}_e_{}.png".format(model, op))
