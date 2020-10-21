import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
perts = ["mlef", "grad"]
fig, ax = plt.subplots()
for pt in perts:
    jh = np.loadtxt("{}_jh_{}_{}.txt".format(model, op, pt))
    x = np.arange(jh.size) + 1
    ax.plot(x, jh, label=pt)
ax.set(xlabel="iteration", ylabel="cost function", title=op)
ax.set_xticks(x[::5])
ax.set_xticks(x, minor=True)
ax.set_xlim(1,20)
ax.legend()
fig.savefig("{}_jh_{}.png".format(model, op))
