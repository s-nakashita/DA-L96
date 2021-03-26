## This program can be run by only Python39
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns

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
    fig = plt.figure(tight_layout=True, figsize=(9,4))
    gs = gridspec.GridSpec(1,2)
    for pt in perts:
        widths = np.array(obs_s)*0.25
    #xaxis = np.array(obs_s) - 3.5*width
        i = 0
        rdata = []
        nrdata = []
        for oberr in oberrs:
            er = []#np.zeros(50)
            enr = []#np.zeros(50)
            jr = 0
            jnr = 0
            for count in range(1, 51):
                f = "cgf_rest_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
            #f = "{}_e_{}_{}_oberr{}.txt".format(model, op, pt, oberr)
                if not os.path.isfile(f):
                    print("not exist {}".format(f))
                    er.append(np.nan)
                else:
                    e = np.loadtxt(f)
                    er.append(np.mean(e[5:]))
                jr += 1
                
                f = "cgf_norest_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
                if not os.path.isfile(f):
                    print("not exist {}".format(f))
                    enr.append(np.nan)
                else:
                    e = np.loadtxt(f)
                    enr.append(np.mean(e[5:]))
                jnr += 1
                #enrs[i] += np.mean(e[5:])**2
            rdata.append(er)
            nrdata.append(enr)
            i += 1

        df_r = pd.DataFrame(np.array(rdata).T, columns=obs_s)
        df_nr = pd.DataFrame(np.array(nrdata).T, columns=obs_s)
        df_r_melt = pd.melt(df_r)
        df_r_melt['setting'] = 'with restart'
        df_nr_melt = pd.melt(df_nr)
        df_nr_melt['setting'] = 'no restart'
        df = pd.concat([df_r_melt, df_nr_melt], axis=0)
        print(df.head())
        #continue
        ind = perts.index(pt)
        ax = fig.add_subplot(gs[ind])
        sns.boxplot(x='variable', y='value', data=df, hue='setting', 
                    showfliers=False, ax=ax)
        #ax1 = fig.add_subplot(gs[ind,0])    
        #ax2 = fig.add_subplot(gs[ind,1])    
        #bplot1 = ax1.boxplot(rdata, vert=True, positions=obs_s, widths=widths,
        #                    patch_artist=True)
        #ax1.set_title(pt+" restart")
        #bplot2 = ax2.boxplot(nrdata, vert=True, positions=obs_s, widths=widths,
        #                    patch_artist=True)
        #ax2.set_title(pt+" no restart")

        #colors = [linecolor[pt]] * len(obs_s)
        #for bplot in (bplot1, bplot2):
        #    for patch, color in zip(bplot['boxes'], colors):
        #        patch.set_facecolor(color)
        #for ax in [ax1, ax2]:
            #ax.plot(obs_s, obs_s, linestyle="dotted", color="tab:gray")
        ax.set(xlabel="observation error", ylabel="RMSE",
                title=op + " " + pt)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        #ax.set_xscale("log")
        ax.set_ylim((1e-5, 5e-1))
        ax.set_yscale("log")
        ax.yaxis.grid(True)
#ax.set_xticks(x[::5])
#ax.set_xticks(x, minor=True)
        ax.legend()
    fig.savefig("{}_ebox_{}_{}.png".format(model, op, method))
    fig.savefig("{}_ebox_{}_{}.pdf".format(model, op, method))
