import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "mlefb", "mleft"]
for i in range(4):
    fig, ax = plt.subplots()
    for pt in perts:
        jh = np.loadtxt("{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, i))
        x = np.arange(jh.size) + 1
        ax.plot(x, jh, label=pt)
    ax.set(xlabel="iteration", ylabel="cost function", title=op)
    ax.set_xticks(x[::5])
    ax.set_xticks(x, minor=True)
    ax.set_xlim(1,20)
    ax.set_yscale("log")
    ax.legend()
    fig.savefig("{}_jh_{}_cycle{}.png".format(model, op, i))
