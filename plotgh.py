import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
perts = ["mlef", "grad"]
fig, ax = plt.subplots()
for pt in perts:
    gh = np.loadtxt("{}_gh_{}_{}.txt".format(model, op, pt))
    x = np.arange(gh.size) + 1
    ax.plot(x, gh, label=pt)
ax.set(xlabel="iteration", ylabel="|g|", title=op)
ax.set_xticks(x[::5])
ax.set_xticks(x, minor=True)
ax.set_xlim(1, 20)
ax.legend()
fig.savefig("{}_gh_{}.png".format(model, op))
