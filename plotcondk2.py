import sys
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
#perts = ["mlef", "grad"]
perts = ["etkf-fh","etkf-jh"]
#if model == "z08":
#    na = 20
#elif model == "l96":
#    na = 100
fig, ax = plt.subplots()
#lags = [4, 6, 8, 10, 12]
obs_s = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
         0.0005, 0.0002, 0.0001]
oberrs = [str(int(obs_s[i]*1e4)).zfill(4) for i in range(len(obs_s))]
print(oberrs)
linecolor={"mlef":"tab:blue","grad":"tab:orange","etkf-fh":"tab:green","etkf-jh":"tab:red"}
linestyle={"etkf-fh":"solid","etkf-jh":"dashed"}
j = 0
for pt in perts:
    i = 0
    el = np.zeros(len(obs_s))
    eld = np.zeros(len(obs_s))
    for oberr in oberrs:
        #f = "{}_e_{}_{}_oberr{}_mean.txt".format(model, op, pt, oberr)
        f = "{}_K2_{}_{}_cycle0_oberr{}.npy".format(model, op, pt, oberr)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            el[i] = np.nan
            i += 1
            continue
        K = np.load(f)
        el[i] = la.cond(K)
        i += 1
    ax.plot(obs_s, el, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    i = 0
    #for oberr in oberrs:
    #    e = np.loadtxt("{}_e_{}-nodiff_{}_oberr{}_mean.txt".format(model, op, pt, oberr))
    #    eld[i] = np.mean(e[5:])
    #    i += 1
    #ax.plot(obs_s, eld, linestyle=linestyle[1], color=linecolor[pt], label="{}-nodiff".format(pt))
    j += 1
#ax.plot(obs_s, obs_s, linestyle="dotted", color="tab:purple")
ax.set(xlabel="observation error", ylabel="2-norm condition number",
        title=op)
#ax.set_ylim(0.0, 0.1)
ax.set_xscale("log")
ax.set_yscale("log")
#ax.set_xticks(x[::5])
#ax.set_xticks(x, minor=True)
ax.legend()
fig.savefig("{}_condk2_{}.png".format(model, op))
#fig.savefig("{}_eoberr_{}+nodiff.pdf".format(model, op))
