import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
perts = ["mlef", "grad"]
#perts = ["mlef","mlefb","mleft"]
#perts = ["etkf-fh","etkf-jh"]
#if model == "z08":
#    na = 20
#elif model == "l96":
#    na = 100
fig, ax = plt.subplots()
#lags = [4, 6, 8, 10, 12]
obs_s = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
         0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
oberrs = [str(int(obs_s[i]*1e5)).zfill(5) for i in range(len(obs_s))]
print(oberrs)
linecolor={"mlef":"tab:blue","grad":"tab:orange","etkf-fh":"tab:green","etkf-jh":"tab:red",
           "mlefb":"tab:cyan","mleft":"tab:pink"}
linestyle=["solid","dashed"]
methods = ["cgf_fr","cgf_pr","cgf_prb"]
#linecolor = {"lb":'tab:blue',"bg":'tab:orange',
#    "cg":'tab:green',"nm":'tab:red',
#    "gd":"tab:purple","cgf_fr":"tab:olive",
#    "cgf_pr":"tab:brown","cgf_prb":"tab:pink"}
j = 0
for method in methods:
    fig, ax = plt.subplots()
    for pt in perts:
    #width = np.array(obs_s)*0.025
    #xaxis = np.array(obs_s) - 3.5*width
        i = 0
        er = np.zeros(len(obs_s))
        ers = np.zeros(len(obs_s))
        enr = np.zeros(len(obs_s))
        enrs = np.zeros(len(obs_s))
        for oberr in oberrs:
            rmean = np.zeros(na+1)
            rstdv = np.zeros(na+1)
            nrmean = np.zeros(na+1)
            nrstdv = np.zeros(na+1)
            jr = 0
            jnr = 0
            for count in range(1, 51):
                f = "cgf_rest_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
            #f = "{}_e_{}_{}_oberr{}.txt".format(model, op, pt, oberr)
                if not os.path.isfile(f):
                    print("not exist {}".format(f))
                    #er[i] += np.nan
                    continue
                e = np.loadtxt(f)
                jr += 1
                rmean += e
                rstdv += e**2
                #er[i] += np.mean(e[5:])
                #ers[i] += np.mean(e[5:])**2
                
                f = "cgf_norest_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
                if not os.path.isfile(f):
                    print("not exist {}".format(f))
                    #enr[i] += np.nan
                    continue
                e = np.loadtxt(f)
                jnr += 1
                nrmean += e
                nrstdv += e**2
                #enr[i] += np.mean(e[5:])
                #enrs[i] += np.mean(e[5:])**2
            rmean /= jr
            nrmean /= jnr
            rstdv = np.sqrt(rstdv/jr - rmean**2)
            nrstdv = np.sqrt(nrstdv/jnr - nrmean**2)
            er[i] = np.mean(rmean[5:])
            ers[i] = np.mean(rstdv[5:])
            enr[i] = np.mean(nrmean[5:])
            enrs[i] = np.mean(nrstdv[5:])
            i += 1
            fm = "cgf_rest_oberr/{}_{}_oberr{}_{}_mean.txt".format(op, pt, oberr, method)
            fs = "cgf_rest_oberr/{}_{}_oberr{}_{}_stdv.txt".format(op, pt, oberr, method)
            np.savetxt(fm, rmean)
            np.savetxt(fs, rstdv)
            fm = "cgf_norest_oberr/{}_{}_oberr{}_{}_mean.txt".format(op, pt, oberr, method)
            fs = "cgf_norest_oberr/{}_{}_oberr{}_{}_stdv.txt".format(op, pt, oberr, method)
            np.savetxt(fm, nrmean)
            np.savetxt(fs, nrstdv)
        print(ers)
        print(enrs)
        #ax.errorbar(obs_s, er, yerr=ers, linestyle=linestyle[0], color=linecolor[pt], label=pt+"-restart")
        ax.plot(obs_s, er, linestyle=linestyle[0], color=linecolor[pt], label=pt+"-restart")
        #ax.errorbar(obs_s, enr, yerr=enrs, linestyle=linestyle[1], color=linecolor[pt], label=pt)
        ax.plot(obs_s, enr, linestyle=linestyle[1], color=linecolor[pt], label=pt)
        
    j += 1
    ax.plot(obs_s, obs_s, linestyle="dotted", color="tab:gray")
    ax.set(xlabel="observation error", ylabel="RMSE",
        title=op+" "+method)
#ax.set_ylim(0.0, 0.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
#ax.set_xticks(x[::5])
#ax.set_xticks(x, minor=True)
    ax.legend()
    fig.savefig("{}_error_{}_{}.png".format(model, op, method))
    fig.savefig("{}_error_{}_{}.pdf".format(model, op, method))
