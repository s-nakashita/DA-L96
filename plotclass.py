import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
perts = ["mlef", "grad"]
#perts = ["mlef","mlefb","mleft"]
#perts = ["etkf-fh","etkf-jh"]
if model == "z08":
    nx = 81
elif model == "l96":
    nx = 40
fig, ax = plt.subplots()
#lags = [4, 6, 8, 10, 12]
#obs_s = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
#         0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
obs_s = [0.05]
oberrs = [str(int(obs_s[i]*1e5)).zfill(5) for i in range(len(obs_s))]
print(oberrs)
#member = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 60, 80]
#print(member)
linecolor={"mlef":"tab:blue","grad":"tab:orange","etkf-fh":"tab:green","etkf-jh":"tab:red",
           "mlefb":"tab:cyan","mleft":"tab:pink"}
linestyle=["solid","dashed"]
#methods = ["cgf_fr","cgf_pr","cgf_prb"]
methods = ["lb"]
#linecolor = {"lb":'tab:blue',"bg":'tab:orange',
#    "cg":'tab:green',"nm":'tab:red',
#    "gd":"tab:purple","cgf_fr":"tab:olive",
#    "cgf_pr":"tab:brown","cgf_prb":"tab:pink"}
j = 0
#nvar = len(obs_s)
#nvar = len(member)
for method in methods:
    for pt in perts:
    #width = np.array(obs_s)*0.025
    #xaxis = np.array(obs_s) - 3.5*width
        i = 0
        emean = dict()
        data = []
        for oberr in oberrs:
            rmean = np.zeros(na+1)
            rstdv = np.zeros(na+1)
            nrmean = np.zeros(na+1)
            nrstdv = np.zeros(na+1)
            jr = 0
            jnr = 0
            for count in range(1, 51):
                f = "lb_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
            #f = "{}_e_{}_{}_oberr{}.txt".format(model, op, pt, oberr)
                if not os.path.isfile(f):
                    print("not exist {}".format(f))
                    #er[i] += np.nan
                    continue
                e = np.loadtxt(f)
                data.append(np.mean(e[5:]))
                emean[np.mean(e[5:])] = count
            print(emean)
            sortedmean = sorted(emean.items())
            print(sortedmean[0][0])
            
            u1 = np.zeros(nx)
            ind1 = [sortedmean[i][1] for i in range(13)]
            u2 = np.zeros(nx)
            ind2 = [sortedmean[i][1] for i in range(13, 25)]
            u3 = np.zeros(nx)
            ind3 = [sortedmean[i][1] for i in range(25, 37)]
            u4 = np.zeros(nx)
            ind4 = [sortedmean[i][1] for i in range(37, 50)]
            umean = np.zeros(nx)
            print(ind1)
            print(ind2)
            print(ind3)
            print(ind4)
            for count in range(1, 51):
                f = "lb_oberr/ua_{}_{}_cycle0_{}.npy".format(op, pt, count)
                if not os.path.isfile(f):
                    print("not exist {}".format(f))
                    continue
                ua = np.load(f)
                if count == ind1[0]:
                    u1 = ua[:, 0]
                if count == ind1[-1]:
                    u2 = ua[:, 0]
                if count == ind3[-1]:
                    u3 = ua[:, 0]
                if count == ind4[-1]:
                    u4 = ua[:, 0]
                if count == ind2[-1] or count==ind3[0]:
                    umean = umean + ua[:, 0]*0.5
            #u1 = u1 / len(ind1)
            #u2 = u2 / len(ind2)
            #u3 = u3 / len(ind3)
            #u4 = u4 / len(ind4)
            f = "lb_oberr/{}_ut.npy".format(model)
            ut = np.load(f)

            x = np.arange(20) + 1
            y = np.zeros_like(x)
            #fig, ax = plt.subplots()
            fig = plt.figure(tight_layout=True, figsize=(9,9))
            gs = gridspec.GridSpec(2,1)

            ax = fig.add_subplot(gs[0])
            ax.plot(x, y, linestyle="dotted", color="gray")
            ax.plot(x, ut[0,:20], color="black", label="true")
            #ax1 = fig.add_subplot(gs[0,0])
            ax.plot(x, u1[:20], label="best")
            #ax1.set_title("best") 
            #ax2 = fig.add_subplot(gs[0,1])
            ax.plot(x, u2[:20], label="Q1")
            #ax2.set_title("better")
            ax.plot(x, umean[:20], label="Q2")
            #ax3 = fig.add_subplot(gs[1,0])
            ax.plot(x, u3[:20], label="Q3")
            #ax3.set_title("worse")
            #ax4 = fig.add_subplot(gs[1,1])
            ax.plot(x, u4[:20], label="worst")
            #ax4.set_title("worst")
            #for ax in [ax1, ax2, ax3, ax4]:
            ax.set(xlabel="point", ylabel="u")
            ax.set_xticks(x[::5])
            ax.set_xticks(x, minor=True)
            ax.set_ylim((-0.5,1.1))
            ax.legend()

            ax2 = fig.add_subplot(gs[1])
            ax2.boxplot(data, vert=False)
            ax2.set_ylabel(obs_s[oberrs.index(oberr)])
            ax2.set_xlabel("RMSE")
            ax2.set_xlim((0.0, 0.1))
            fig.savefig("{}_class_{}_{}_oberr{}.png".format(model, op, pt, oberr))