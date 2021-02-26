import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "grad", "mlefb", "mleft", "mlef05", "mlef3", "mlefw", "mlefh"]
linestyle = ["solid", "dashed"]
for i in range(4):
    fig, ax = plt.subplots()
    for pt in perts:
        f = "{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, i)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        gh = np.loadtxt(f)
        x = np.arange(gh.size) + 1
        jj = perts.index(pt)
        j = jj - int(jj/2)*2
        ax.plot(x, gh, linestyle=linestyle[j], label=pt)
    ax.set(xlabel="iteration", ylabel="|g|", title=op)
    ax.set_xticks(x[::5])
    ax.set_xticks(x, minor=True)
    ax.set_xlim(1, 20)
    ax.set_yscale("log")
    ax.legend()
    fig.savefig("{}_gh_{}_cycle{}.png".format(model, op, i))
