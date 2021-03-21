import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "grad", "mlefb", "mleft", "mlef05", "grad05", 
         "mlefw", "mlef3", "mlefh"]
linestyle = ["solid", "dashed"]
for i in range(4):
    fig, ax = plt.subplots()
    lenx = []
    for pt in perts:
        f = "{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, i)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        jh = np.loadtxt(f)
        x = np.arange(jh.size) + 1
        lenx.append(x.size)
        jj = perts.index(pt)
        j = jj - int(jj/2)*2
        ax.plot(x, jh, linestyle=linestyle[j], label=pt)
    ax.set(xlabel="iteration", ylabel="cost function", title=op)
    xaxis = np.arange(np.max(lenx)) + 1
    ax.set_xticks(xaxis[::5])
    ax.set_xticks(xaxis, minor=True)
    ax.set_xlim(1,20)
    ax.set_yscale("log")
    ax.legend()
    fig.savefig("{}_jh_{}_cycle{}.png".format(model, op, i))
