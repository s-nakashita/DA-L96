import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
perts = ["mlefw", "gradw"]
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
#member = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40]
#print(member)
linecolor0={"mlef":"tab:blue","grad":"tab:orange","etkf-fh":"tab:green","etkf-jh":"tab:red",
           "mlefw":"tab:blue","gradw":"tab:orange","mlefb":"tab:cyan","mleft":"tab:pink"}
linestyle0=["solid","dashed"]
methods = ["ncg", "tnc"]
methods = ["dog", "trn", "trk", "tre"]
#methods = ["cgf_fr", "cgf_pr", "cgf_prb"]
#methods = ["bg", "nm", "pw" ,"gd", "gdf", 
#"ncg", "tnc", "dog"]
linestyle = {"lb":'solid',"bg":'dashed',
    "pw":'solid',"nm":'solid',
    "gd":"solid", "gdf":"dashed",
    "ncg":"solid", "tnc":"dashdot",
    "dog":"solid","cgf_fr":"solid",
    "cgf_pr":"dashed","cgf_prb":"dashdot"}
hatch = {"lb":'',"bg":'/',
    "pw":'',"nm":'',
    "gd":"", "gdf":"/",
    "ncg":"", "tnc":"/",
    "dog":"","cgf_fr":"",
    "cgf_pr":"/","cgf_prb":"//"}
linecolor = {"lb":'tab:blue',"bg":'tab:blue',
    "pw":'tab:green',"nm":'tab:red',
    "gd":"tab:purple","gdf":"tab:purple",
    "ncg":"tab:pink","tnc":"tab:pink",
    "dog":"tab:olive",
    "cgf_fr":"orange",
    "cgf_pr":"orange","cgf_prb":"orange"}
j = 0
nvar = len(obs_s)
#nvar = len(member)
for method in methods:
#for pt in perts:
    fig, ax = plt.subplots()
    #fig = plt.figure()
    #ax = fig.subplot_mosaic()
    for pt in perts:
#    width = np.array(obs_s)*0.05
#    xaxis = np.array(obs_s) - 6.0*width    
#    for method in methods:
        i = 0
        er = np.zeros(nvar)
        ers = np.zeros(nvar)
        enr = np.zeros(nvar)
        enrs = np.zeros(nvar)
        for oberr in oberrs:
        #for nmem in member:
            rmean = np.zeros(na+1)
            rstdv = np.zeros(na+1)
            nrmean = np.zeros(na+1)
            nrstdv = np.zeros(na+1)
            jr = 0
            jnr = 0
            for count in range(1, 51):
                #f = "cgf_rest_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
                #f = "lb_rest2_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
                f = "ncg_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
                #f = "mlef-rest_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
                #f = "lb_rest_nmem/{}_{}_nmem{}_{}_{}.txt".format(op, pt, nmem, method, count)
                #f = "etkf_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
            #f = "{}_e_{}_{}_oberr{}.txt".format(model, op, pt, oberr)
                if not os.path.isfile(f):
                    print("not exist {}".format(f))
                    rmean += np.nan
                    continue
                e = np.loadtxt(f)
                jr += 1
                rmean += e
                rstdv += e**2
                #er[i] += np.mean(e[5:])
                #ers[i] += np.mean(e[5:])**2
                
#                #f = "cgf_norest_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
#                f = "lb_norest_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
#                #f = "mlef-nd_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
#                #f = "lb_nmem/{}_{}_nmem{}_{}_{}.txt".format(op, pt, nmem, method, count)
#                #f = "etkf-fh_oberr/{}_{}_oberr{}_{}_{}.txt".format(op, pt, oberr, method, count)
#                if not os.path.isfile(f):
#                    print("not exist {}".format(f))
#                    nrmean += np.nan
#                    continue
#                e = np.loadtxt(f)
#                jnr += 1
#                nrmean += e
#                nrstdv += e**2
#                #enr[i] += np.mean(e[5:])
#                #enrs[i] += np.mean(e[5:])**2
            rmean /= jr
#            nrmean /= jnr
            rstdv = np.sqrt(rstdv/jr - rmean**2)
#            nrstdv = np.sqrt(nrstdv/jnr - nrmean**2)
            er[i] = np.mean(rmean[5:])
            ers[i] = np.mean(rstdv[5:])
#            enr[i] = np.mean(nrmean[5:])
#            enrs[i] = np.mean(nrstdv[5:])
            i += 1
            #fm = "cgf_rest_oberr/{}_{}_oberr{}_{}_mean.txt".format(op, pt, oberr, method)
            #fm = "lb_rest2_oberr/{}_{}_oberr{}_{}_mean.txt".format(op, pt, oberr, method)
            #fm = "mlef-rest_oberr/{}_{}_oberr{}_{}_mean.txt".format(op, pt, oberr, method)
            fm = "ncg_oberr/{}_{}_oberr{}_{}_mean.txt".format(op, pt, oberr, method)
            #fm = "lb_rest_nmem/{}_{}_nmem{}_{}_mean.txt".format(op, pt, nmem, method)
            #fm = "etkf_oberr/{}_{}_oberr{}_mean.txt".format(op, pt, oberr)
            #fs = "cgf_rest_oberr/{}_{}_oberr{}_{}_stdv.txt".format(op, pt, oberr, method)
            #fs = "lb_rest2_oberr/{}_{}_oberr{}_{}_stdv.txt".format(op, pt, oberr, method)
            #fs = "mlef-rest_oberr/{}_{}_oberr{}_{}_stdv.txt".format(op, pt, oberr, method)
            fs = "ncg_oberr/{}_{}_oberr{}_{}_stdv.txt".format(op, pt, oberr, method)
            #fs = "lb_rest_nmem/{}_{}_nmem{}_{}_stdv.txt".format(op, pt, nmem, method)
            #fs = "etkf_oberr/{}_{}_oberr{}_stdv.txt".format(op, pt, oberr)
            np.savetxt(fm, rmean)
            np.savetxt(fs, rstdv)
#            #fm = "cgf_norest_oberr/{}_{}_oberr{}_{}_mean.txt".format(op, pt, oberr, method)
#            #fm = "lb_norest_oberr/{}_{}_oberr{}_{}_mean.txt".format(op, pt, oberr, method)
#            fm = "mlef-nd_oberr/{}_{}_oberr{}_{}_mean.txt".format(op, pt, oberr, method)
#            #fm = "lb_nmem/{}_{}_nmem{}_{}_mean.txt".format(op, pt, nmem, method)
#            #fm = "etkf-fh_oberr/{}_{}_oberr{}_mean.txt".format(op, pt, oberr)
#            #fs = "cgf_norest_oberr/{}_{}_oberr{}_{}_stdv.txt".format(op, pt, oberr, method)
#            #fs = "lb_norest_oberr/{}_{}_oberr{}_{}_stdv.txt".format(op, pt, oberr, method)
#            fs = "mlef-nd_oberr/{}_{}_oberr{}_{}_stdv.txt".format(op, pt, oberr, method)
#            #fs = "lb_nmem/{}_{}_nmem{}_{}_stdv.txt".format(op, pt, nmem, method)
#            #fs = "etkf-fh_oberr/{}_{}_oberr{}_stdv.txt".format(op, pt, oberr)
#            np.savetxt(fm, nrmean)
#            np.savetxt(fs, nrstdv)
        print(ers)
#        print(enrs)
        #ax.errorbar(obs_s, er, yerr=ers, linestyle=linestyle[0], color=linecolor[pt], label=pt+"-restart")
        #if pt == "etkf-fh":
        #    label = pt + " av-op"
        #else:
        label = pt #+ "-restart"
        ax.plot(obs_s, er, linestyle=linestyle0[0], color=linecolor0[pt], label=label)
        #ax.plot(obs_s, er, linestyle=linestyle[method], color=linecolor[method], label=label)
        #ax.bar(xaxis, er, width, hatch=hatch[method], color=linecolor[method], label=label)
        #xaxis = xaxis + width
        #ax.plot(member, er, linestyle=linestyle[0], color=linecolor[pt], label=pt)
        #ax.errorbar(obs_s, enr, yerr=enrs, linestyle=linestyle[1], color=linecolor[pt], label=pt)
        #if pt == "etkf-fh":
        #    label = pt + " op-av"
#        label = pt
#        ax.plot(obs_s, enr, linestyle=linestyle0[1], color=linecolor0[pt], label=label)
        #ax.plot(member, enr, linestyle=linestyle[1], color=linecolor[pt], label=pt)
        
    j += 1
    ax.plot(obs_s, obs_s, linestyle="dotted", color="tab:gray")
    ax.set(xlabel="observation error", ylabel="RMSE",
        title=op+" "+method)
#    y = np.ones(nvar) * 0.001
#    ax.plot(member, y, linestyle="dotted", color="tab:gray")
#    ax.set(xlabel="ensemble size", ylabel="RMSE",
#        title=op)
#ax.set_ylim(0.0, 0.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    #ax.set_xticks(member)
#ax.set_xticks(x[::5])
#ax.set_xticks(x, minor=True)
    ax.legend()#ncol=3)
    fig.savefig("{}_error_{}_{}w.png".format(model, op, method))
    fig.savefig("{}_error_{}_{}w.pdf".format(model, op, method))
