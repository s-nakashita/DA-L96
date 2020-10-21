import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
if model == "z08":
    nx = 81
elif model == "l96":
    nx = 40
dh_mlef = np.load("{}_dh_{}_{}.npy".format(model, op, "mlef"))
dh_grad = np.load("{}_dh_{}_{}.npy".format(model, op, "grad"))
x = np.arange(nx) + 1
fig, ax = plt.subplots(2,2)
for j in range(2):
    for i in range(2):
        ax[j, i].plot(x, dh_mlef[:, 2*j+i], label="mlef")
        ax[j, i].plot(x, dh_grad[:, 2*j+i], label="grad")
        ax[j, i].set_xticks(x[::10])
        ax[j, i].set_xticks(x[::5], minor=True)
ax[0, 0].legend()
fig.suptitle(op)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("{}_dh_{}.png".format(model, op))
