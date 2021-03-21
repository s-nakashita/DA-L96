import sys
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = sys.argv[4]
if model == "z08" or model == "z05":
    nx = 81
    #na = 20
elif model == "l96":
    nx = 40
    #na = 300
x = np.arange(nx+1) + 1
cmap = "Reds"

icycle = 0
f = "{}_tmat_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit()
tmat = np.load(f)

f = "{}_pf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit()
pf = np.load(f)

f = "{}_gmat_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit()
gmat = np.load(f)
    
nmem = tmat.shape[0]
mode = np.arange(nmem)+1
site = np.arange(nx)
u, s, vt = la.svd(tmat)
v = vt.transpose()
print(s)
contr = s / np.sum(s)

fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2,2)

ax = fig.add_subplot(gs[0,0])
ax2 = ax.twinx()
for i in range(nmem):
    ax.bar(mode[i], s[i], width=0.35)
ax2.plot(mode, contr, color="purple", label="contribution")
ax2.legend()
ax2.tick_params(axis='y', colors="purple")
ax2.set_ylim([0.0,1.0])
ax.set(title=r"singular values $\Sigma$ of $[I+C]^{-1/2}$", 
            xlabel="mode")
ax.set_xticks(mode)
ax.set_yscale("log")

ax = fig.add_subplot(gs[0,1])
width=0.2
xaxis = mode - width*1.5
for i in range(nmem):
    ax.bar(xaxis,v[:,i],width=width, label="mode{}".format(i+1))
    xaxis += width
ax.set_xticks(mode)
ax.set(title=r"left singular vectors V of $[I+C]^{-1/2}$",
            xlabel="zeta")
#ax.legend()

sv = np.diag(s) @ vt
ax = fig.add_subplot(gs[1,0])
width=0.2
xaxis = mode - width*1.5
for i in range(nmem):
    ax.bar(xaxis,sv[i],width=width, label="mode{}".format(i+1))
    xaxis += width
ax.set_xticks(mode)
ax.set(title=r"$\Sigma V^T$",
            xlabel="zeta")
fig.align_labels()
fig.savefig("{}_tmat1_{}_{}.png".format(model,op,pt))

fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2,2)

ax = fig.add_subplot(gs[0,0])
width=0.2
xaxis = mode - width*1.5
for i in range(nmem):
    ax.bar(xaxis,u[:,i],width=width, label="mode{}".format(i+1))
    xaxis += width
ax.set_xticks(mode)
ax.set(title=r"right singular vectors U of $[I+C]^{-1/2}$",
            xlabel="member")

ax = fig.add_subplot(gs[1,:])
pfu = pf @ u
for i in range(nmem):
    ax.plot(site,pfu[:,i], label="mode{}".format(i+1))
ax.set_xticks(site[::10])
ax.set(title=r"$P_f^{1/2}U$", 
            xlabel="point")
ax.legend(ncol=2)
fig.align_labels()
fig.savefig("{}_tmat2_{}_{}.png".format(model,op,pt))

fig, ax = plt.subplots()
for i in range(nmem):
    ax.plot(site,gmat[:,i], label="zeta{}".format(i+1))
ax.set_xticks(site[::10])
ax.set(title=r"$P_f^{1/2}[I+C]^{-1/2}$", 
            xlabel="point")
ax.legend(ncol=2)
fig.savefig("{}_gmat_{}_{}.png".format(model,op,pt))