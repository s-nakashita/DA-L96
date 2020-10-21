import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
if model == "z08":
    nx = 81
    na = 20
elif model == "l96":
    nx = 40
    na = 100
ut = np.load("ut.npy")[0:4,:]
du_mlef = np.load("{}_ua_{}_{}.npy".format(model, op, "mlef"))[0:4,:,0] - ut
du_grad = np.load("{}_ua_{}_{}.npy".format(model, op, "grad"))[0:4,:,0] - ut
x = np.arange(nx) + 1
fig, ax = plt.subplots(2,2)
for j in range(2):
    for i in range(2):
        ax[j, i].plot(x, du_mlef[2*j+i, :], label="mlef")
        ax[j, i].plot(x, du_grad[2*j+i, :], label="grad")
        ax[j, i].set_xticks(x[::10])
        ax[j, i].set_xticks(x[::5], minor=True)
ax[0, 0].legend()
fig.suptitle(op)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("{}_du_{}.png".format(model,op))
