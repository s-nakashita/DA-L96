import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
if model == "z08":
    nx = 81
elif model == "l96":
    nx = 40
cJ_mlef = np.load("{}_cJ_{}_{}.npy".format(model, op, "mlef"))
cJ_grad = np.load("{}_cJ_{}_{}.npy".format(model, op, "grad"))
x = np.linspace(-1000,1000,cJ_mlef.shape[0])
fig, ax = plt.subplots(2,2)
for j in range(2):
    for i in range(2):
        ax[j, i].plot(x, cJ_mlef[:, 2*j+i], label="mlef")
        ax[j, i].plot(x, cJ_grad[:, 2*j+i], label="grad")
        #ax[j, i].set_xticks(x[::200])
        #ax[j, i].set_xticks(x[::100], minor=True)
        ax[j, i].set_title("mem {}".format(2*j+i+1))
ax[0, 0].legend()
fig.suptitle(op)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("{}_cJ_{}.png".format(model, op))
