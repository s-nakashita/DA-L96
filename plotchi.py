import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
#perts = ["etkf", "po", "srf", "letkf"]
perts = ["mlef", "grad"]
if model == "z08":
    na = 20
elif model == "l96":
    na = 100
x = np.arange(na) + 1
fig, ax = plt.subplots()
for pt in perts:
    chi = np.loadtxt("{}_chi_{}_{}.txt".format(model, op, pt))
    ax.plot(x, chi, label=pt)
ax.set(xlabel="analysis cycle", ylabel="Chi2",
        title=op)
#ax.set_xticks(x[::5])
#ax.set_xticks(x, minor=True)
ax.legend()
fig.savefig("{}_chi_{}.png".format(model, op))
