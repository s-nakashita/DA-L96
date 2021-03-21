import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "grad"]
linestyle = ["solid", "dashed"]
for i in range(4):
    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(3, 2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    lenx = []
    #for pt in perts:
    pt = perts[0]
    f = "{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, i)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    jh = np.loadtxt(f)
    x = np.arange(jh.size) + 1
    lenx.append(x.size)
    jj = perts.index(pt)
    j = jj - int(jj/2)*2
    ax00.plot(x, jh, linestyle=linestyle[j], label=pt)
    diff = np.log10(jh.max())-np.log10(jh.min())
    if diff > 1:
        ax00.set_yscale("log")

    f = "{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, i)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    gh = np.loadtxt(f)
    x = np.arange(gh.size) + 1
    #lenx.append(x.size)
    jj = perts.index(pt)
    j = jj - int(jj/2)*2
    ax10.plot(x, gh, linestyle=linestyle[j], label=pt)
    diff = np.log10(gh.max())-np.log10(gh.min())
    if diff > 1:
        ax10.set_yscale("log")

    f = "{}_alpha_{}_{}_cycle{}.txt".format(model, op, pt, i)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    alpha = np.loadtxt(f)
    x = np.arange(alpha.size) + 1
    jj = perts.index(pt)
    j = jj - int(jj/2)*2
    ax20.plot(x, alpha, linestyle=linestyle[j], label=pt)

    pt = perts[1]
    f = "{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, i)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    jh = np.loadtxt(f)
    x = np.arange(jh.size) + 1
    lenx.append(x.size)
    jj = perts.index(pt)
    j = jj - int(jj/2)*2
    ax01.plot(x, jh, linestyle=linestyle[j], label=pt)
    diff = np.log10(jh.max())-np.log10(jh.min())
    if diff > 1:
        ax01.set_yscale("log")
    
    f = "{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, i)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    gh = np.loadtxt(f)
    x = np.arange(gh.size) + 1
    #lenx.append(x.size)
    jj = perts.index(pt)
    j = jj - int(jj/2)*2
    ax11.plot(x, gh, linestyle=linestyle[j], label=pt)
    diff = np.log10(gh.max())-np.log10(gh.min())
    if diff > 1:
        ax11.set_yscale("log")

    f = "{}_alpha_{}_{}_cycle{}.txt".format(model, op, pt, i)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    alpha = np.loadtxt(f)
    x = np.arange(alpha.size) + 1
    jj = perts.index(pt)
    j = jj - int(jj/2)*2
    ax21.plot(x, alpha, linestyle=linestyle[j], label=pt)
    
    ax00.set(xlabel="iteration", ylabel="cost function", title=op)
    ax01.set(xlabel="iteration", ylabel="cost function", title=op)
    ax10.set(xlabel="iteration", ylabel="|g|", title=op)
    ax11.set(xlabel="iteration", ylabel="|g|", title=op)
    ax20.set(xlabel="iteration", ylabel="alpha", title=op)
    ax21.set(xlabel="iteration", ylabel="alpha", title=op)
    xaxis = np.arange(np.max(lenx)) + 1
    if len(xaxis) > 100:
        ax00.set_xticks(xaxis[::50])
        ax00.set_xticks(xaxis[::10], minor=True)
        ax01.set_xticks(xaxis[::50])
        ax01.set_xticks(xaxis[::10], minor=True)
        ax10.set_xticks(xaxis[::50])
        ax10.set_xticks(xaxis[::10], minor=True)
        ax11.set_xticks(xaxis[::50])
        ax11.set_xticks(xaxis[::10], minor=True)
        ax20.set_xticks(xaxis[::50])
        ax20.set_xticks(xaxis[::10], minor=True)
        ax21.set_xticks(xaxis[::50])
        ax21.set_xticks(xaxis[::10], minor=True)
    elif len(xaxis) > 100:
        ax00.set_xticks(xaxis[::20])
        ax00.set_xticks(xaxis[::5], minor=True)
        ax01.set_xticks(xaxis[::20])
        ax01.set_xticks(xaxis[::5], minor=True)
        ax10.set_xticks(xaxis[::20])
        ax10.set_xticks(xaxis[::5], minor=True)
        ax11.set_xticks(xaxis[::20])
        ax11.set_xticks(xaxis[::5], minor=True)
        ax20.set_xticks(xaxis[::20])
        ax20.set_xticks(xaxis[::5], minor=True)
        ax21.set_xticks(xaxis[::20])
        ax21.set_xticks(xaxis[::5], minor=True)
    else:
        ax00.set_xticks(xaxis[::5])
        ax00.set_xticks(xaxis, minor=True)
        ax01.set_xticks(xaxis[::5])
        ax01.set_xticks(xaxis, minor=True)
        ax10.set_xticks(xaxis[::5])
        ax10.set_xticks(xaxis, minor=True)
        ax11.set_xticks(xaxis[::5])
        ax11.set_xticks(xaxis, minor=True)
        ax20.set_xticks(xaxis[::5])
        ax20.set_xticks(xaxis, minor=True)
        ax21.set_xticks(xaxis[::5])
        ax21.set_xticks(xaxis, minor=True)
    ax00.legend()
    ax01.legend()
    ax10.legend()
    ax11.legend()
    ax20.legend()
    ax21.legend()
    fig.tight_layout()
    fig.savefig("{}_gh_{}_cycle{}.png".format(model, op, i))
