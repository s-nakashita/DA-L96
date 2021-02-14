import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
if model == "z08" or model == "z05":
    nx = 81
    na = 20
elif model == "l96":
    nx = 40
    na = 100
pa_mlef = np.load("{}_pa_{}_{}.npy".format(model, op, "mlef"))[0:4,:,:]
pa_grad = np.load("{}_pa_{}_{}.npy".format(model, op, "grad"))[0:4,:,:]
ymax = max(np.max(pa_mlef),np.max(pa_grad))
ylim = 10**(int(np.log10(ymax)))
if (ylim/2 - ymax > 0):
    ylim = ylim / 2
print(ylim)
x = np.arange(nx) + 1
for j in range(4):
    fig, ax = plt.subplots(1,2)
    for k in range(pa_mlef.shape[2]):
        ax[0].plot(x, pa_mlef[j, :, k], label="member{}".format(k+1))
        ax[0].set_xticks(x[::10])
        ax[0].set_xticks(x[::5], minor=True)
    for k in range(pa_grad.shape[2]):
        ax[1].plot(x, pa_grad[j, :, k], label="member{}".format(k+1))
        ax[1].set_xticks(x[::10])
        ax[1].set_xticks(x[::5], minor=True)
    ax[0].set_ylim(-ylim,ylim)
    ax[0].legend()
    ax[0].set_title("mlef")
    ax[1].set_ylim(-ylim,ylim)
    ax[1].legend()
    ax[1].set_title("grad")
    fig.suptitle("{} cycle{}".format(op, j+1))
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig("{}_sqrtpa_{}_cycle{}.png".format(model,op,j+1))
