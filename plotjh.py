import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "grad"] #, "mlefb", "mleft", "mlef05", "grad05", 
         #"mlefw", "mlef3", "mlefh"]
linestyle = ["solid", "dashed"]
linecolor = {"mlef":"blue", "grad":"orange"}
plt.rcParams['legend.fontsize'] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 16
for i in range(1):
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
        if pt == "grad":
            label = "mlef-jh"
        elif pt == "mlef":
            label = "mlef-fh"
        ax.plot(x, jh, linestyle=linestyle[0], color="tab:"+linecolor[pt], label=label)
        #f = "cgf_fr-rest/{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, i)
        #if not os.path.isfile(f):
        #    print("not exist {}".format(f))
        #    continue
        #jh = np.loadtxt(f)
        #x = np.arange(jh.size) + 1
        #lenx.append(x.size)
        #jj = perts.index(pt)
        #j = jj - int(jj/2)*2
        #if pt == "grad":
        #    label = "mlef-jh+restart"
        #elif pt == "mlef":
        #    label = "mlef-fh+restart"
        #ax.plot(x, jh, linestyle=linestyle[0], color=linecolor[pt], label=label)
    ax.set(xlabel="iteration", title=r"$J$")#, title=op)
    xaxis = np.arange(np.max(lenx)) + 1
    ax.set_xticks(xaxis[::5])
    ax.set_xticks(xaxis, minor=True)
    ax.set_xlim(1,20)
    ax.set_yscale("log")
    ax.legend()
    fig.savefig("{}_jh_{}_first.pdf".format(model, op))

    fig, ax = plt.subplots()
    lenx = []
    for pt in perts:
        f = "{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, i)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        gh = np.loadtxt(f)
        x = np.arange(gh.size) + 1
        lenx.append(x.size)
        jj = perts.index(pt)
        j = jj - int(jj/2)*2
        if pt == "grad":
            label = "mlef-jh"
        elif pt == "mlef":
            label = "mlef-fh"
        ax.plot(x, gh, linestyle=linestyle[0], color="tab:"+linecolor[pt], label=label)
        
        #f = "cgf_fr-rest/{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, i)
        #if not os.path.isfile(f):
        #    print("not exist {}".format(f))
        #    continue
        #gh = np.loadtxt(f)
        #x = np.arange(gh.size) + 1
        #lenx.append(x.size)
        #jj = perts.index(pt)
        #j = jj - int(jj/2)*2
        #if pt == "grad":
        #    label = "mlef-jh+restart"
        #elif pt == "mlef":
        #    label = "mlef-fh+restart"
        #ax.plot(x, gh, linestyle=linestyle[0], color=linecolor[pt], label=label)
    ax.set(xlabel="iteration", title=r"$\nabla J$")
    xaxis = np.arange(np.max(lenx)) + 1
    ax.set_xticks(xaxis[::5])
    ax.set_xticks(xaxis, minor=True)
    ax.set_xlim(1,20)
    ax.set_yscale("log")
    ax.legend()
    fig.savefig("{}_gh_{}_first.pdf".format(model, op))
