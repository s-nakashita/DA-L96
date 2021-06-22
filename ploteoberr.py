import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]
#perts = ["mlef", "etkf-fh"]
#perts = ["grad", "etkf-jh"]
#if model == "z08":
#    na = 20
#elif model == "l96":
#    na = 100
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
#plt.rcParams['xtick.labelsize'] = 16
#plt.rcParams['ytick.labelsize'] = 16
fig, ax = plt.subplots()
#lags = [4, 6, 8, 10, 12]
obs_s = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
         0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
oberrs = [str(int(obs_s[i]*1e5)).zfill(5) for i in range(len(obs_s))]
print(oberrs)
linecolor={"mlef":"tab:blue","grad":"tab:orange","etkf-fh":"tab:green","etkf-jh":"tab:red",
           "mlefb":"tab:cyan","mleft":"tab:pink"}
linestyle=["solid","dashed","dashdot"]
methods = ["lb","bg","cg","nm","gd"]#,"cgf_fr","cgf_pr","cgf_prb"]
#linecolor = {"lb":'tab:blue',"bg":'tab:orange',
#    "cg":'tab:green',"nm":'tab:red',
#    "gd":"tab:purple","cgf_fr":"tab:olive",
#    "cgf_pr":"tab:brown","cgf_prb":"tab:pink"}
# mlef_oberr
#shed_q = [0,0,0,3,3,3,3,3,0,3,3,3,3,3,3]
#shed_c = [1,2,0,3,3,3,3,3,3,3,3,3,3,3,3]
#shed_qr = [0,3,0,3,2,3,3,3,3,3,3,3,3,3,3]
#shed_qd = [1,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
#shed_cd = [0,3,0,3,3,3,3,0,3,0,3,3,3,3,3]
#shed_qrd = [2,3,0,3,3,3,3,2,3,0,3,3,3,3,3]
# lb_rest2
shed_q = [3,3,2,3,2,3,3,3,3,3,3,3,3,3,3]
shed_c = [3,3,3,3,3,3,0,0,3,3,3,3,3,3,3]
shed_qr = [3,3,3,2,3,3,3,3,3,3,3,3,3,3,3]
shed_qd = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
shed_cd = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
shed_qrd = [3,3,3,3,3,3,2,3,3,0,3,3,3,3,2]
shed = {"quadratic":shed_q,"cubic":shed_c,"quartic":shed_qr,
        "quadratic-nodiff":shed_qd,"cubic-nodiff":shed_cd,"quartic-nodiff":shed_qrd}
j = 0
for pt in perts:
    #fig, ax = plt.subplots()
    #width = np.array(obs_s)*0.025
    #xaxis = np.array(obs_s) - 3.5*width
    #for method in methods:
    i = 0
    el = np.zeros(len(obs_s))
    eld = np.zeros(len(obs_s))
    for oberr in oberrs:
        #f = "{}_e_{}_{}_oberr{}_mean.txt".format(model, op, pt, oberr)
        if pt[0:4] == "etkf":
            f = "etkf_oberr/{}_{}_oberr{}_mean.txt".format(op, pt, oberr)
        else:
            #f = "mlef_oberr/{}_{}_oberr{}_mean.txt".format(op, pt, oberr)
            f = "cgf_norest_oberr/{}_{}_oberr{}_cgf_fr_mean.txt".format(op, pt, oberr)
            #f = "lb_rest2_oberr/{}_{}_oberr{}_lb_mean.txt".format(op, pt, oberr)
            #f = "{}_e_{}_{}_oberr{}_{}.txt".format(model, op, pt, oberr, method)
        #f = "{}_e_{}_{}_oberr{}.txt".format(model, op, pt, oberr)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            el[i] = np.nan
            i += 1
            continue
        e = np.loadtxt(f)
        el[i] = np.mean(e[5:])
        i += 1
    if pt == "etkf-fh":
        label = pt + " av-op"
    elif pt == "grad":
        label = "mlef-jh"
    elif pt == "mlef":
        label = "mlef-fh"
    else:
        label = pt
    ax.plot(obs_s, el, linestyle=linestyle[0], color=linecolor[pt], label=label)
    #ax.plot(obs_s, el, linestyle=linestyle[j], color="tab:blue", label=label)
    #    ax.plot(obs_s, el, linestyle=linestyle[0], color=linecolor[method], label=method)
        #ax.bar(xaxis, el, width=width, color=linecolor[method], label=method)
        #xaxis += width
  
    i = 0
    if pt == "etkf-fh":
        j += 1
        for oberr in oberrs:
        #f = "{}_e_{}-nodiff_{}_oberr{}_mean.txt".format(model, op, pt, oberr)
            f = "etkf-fh_oberr/{}_{}_oberr{}_mean.txt".format(op, pt, oberr)
        #    f = "{}_e_{}-nodiff_{}_oberr{}_{}.txt".format(model, op, pt, oberr, method)
            if not os.path.isfile(f):
                print("not exist {}".format(f))
                eld[i] = np.nan
                i += 1
                continue
            e = np.loadtxt(f)
            eld[i] = np.mean(e[5:])
            i += 1
        label = pt + " op-av"
        ax.plot(obs_s, eld, linestyle=linestyle[1], color=linecolor[pt], label=label)
        #ax.plot(obs_s, eld, linestyle=linestyle[j], color="tab:blue", label=label)
    #    ax.plot(obs_s, eld, linestyle=linestyle[1], color=linecolor[method], label="{}-nodiff".format(method))
    
    j += 1
ax.plot(obs_s, obs_s, linestyle="dotted", color="tab:gray")
#thres = np.array(shed[op])
#print(thres < 3)
#print(thres >= 2)
#ax.fill_between(obs_s, 0, 1, where=thres >= 3, 
#        color="red", alpha=0.3, transform=ax.get_xaxis_transform())
#ax.fill_between(obs_s, 0, 1, where=(np.array(2 <= thres) & np.array(thres < 3)), 
#        color="blue", alpha=0.3, transform=ax.get_xaxis_transform())
#ax.fill_between(obs_s, 0, 1, where=(np.array(1 <= thres) & np.array(thres < 2)), 
#        color="green", alpha=0.3, transform=ax.get_xaxis_transform())
ax.set(xlabel="observation error", ylabel="RMSE",
        title=op)
#ax.set_ylim(0.0, 0.1)
ax.set_xscale("log")
ax.set_yscale("log")
#ax.set_xticks(x[::5])
#ax.set_xticks(x, minor=True)
#ax.legend(ncol=2)
#fig.savefig("{}_eoberr_{}_mlef.png".format(model, op))
fig.savefig("{}_eoberr_{}.pdf".format(model, op))
