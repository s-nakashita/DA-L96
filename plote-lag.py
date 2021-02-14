import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
perts = ["mlef", "grad"]
if model == "z08":
    sigma = {"linear": 8.0e-2, "quadratic": 8.0e-2, "cubic": 7.0e-4, \
    "quadratic-nodiff": 8.0e-2, "cubic-nodiff": 7.0e-4,}
elif model == "l96":
    sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0}
fig, ax = plt.subplots()
lags = [4, 6, 8, 10, 12, 14, 16, 18]
el = np.zeros(len(lags))
linestyle=["solid","dashed"]
j = 0
for pt in perts:
    i = 0
    for lag in lags:
        e = np.loadtxt("{}_e_{}_{}_lag{}_mean.txt".format(model, op, pt, lag))
        el[i] = np.mean(e[5:])
        i += 1
    ax.plot(lags, el, linestyle=linestyle[j], label=pt)
    j += 1
obs_s = np.ones(len(lags)) * sigma[op]
ax.plot(lags, obs_s, linestyle="dotted")
ax.set(xlabel="initial lag", ylabel="RMSE",
        title=op)
#ax.set_ylim(0.0, 0.1)
#ax.set_xscale("log")
#ax.set_yscale("log")
#ax.set_xticks(x[::5])
#ax.set_xticks(x, minor=True)
ax.legend()
fig.savefig("{}_e-lag_{}.png".format(model, op))
