import sys
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
perts = ["mlef", "grad"]
if model == "z08":
    na = 20
elif model == "l96":
    na = 100
fig, ax = plt.subplots()
#lags = [4, 6, 8, 10, 12]
obs_s = [0.1, 0.01, 0.001, 0.0001]
oberrs = [-int(np.log10(obs_s[i])) for i in range(len(obs_s))]
print(oberrs)
el = np.zeros(len(obs_s))
linestyle=["solid","dashed"]
j = 0
for pt in perts:
    i = 0
    for oberr in oberrs:
        e = np.loadtxt("{}_e_{}_{}_oberr{}.txt".format(model, op, pt, oberr))
        el[i] = np.mean(e[5:])
        i += 1
    ax.plot(obs_s, el, linestyle=linestyle[j], label=pt)
    j += 1
ax.plot(obs_s, obs_s, linestyle="dotted")
ax.set(xlabel="observation error", ylabel="RMSE",
        title=op)
#ax.set_ylim(0.0, 0.1)
ax.set_xscale("log")
ax.set_yscale("log")
#ax.set_xticks(x[::5])
#ax.set_xticks(x, minor=True)
ax.legend()
fig.savefig("{}_eoberr_{}.png".format(model, op))
