import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "mlefb"]
fig, ax = plt.subplots()
for pt in perts:
    cg = np.loadtxt("{}_checkg_{}_{}.txt".format(model, op, pt))
    x = np.arange(cg.size) + 1
    ax.plot(x, cg, label=pt)
ax.set(xlabel="cycle", ylabel="err", title=op)
ax.set_xticks(x[::5])
ax.set_xticks(x, minor=True)
#ax.set_xlim(1,20)
ax.set_yscale("log")
ax.legend()
fig.savefig("{}_cg_{}.png".format(model, op))
