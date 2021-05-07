import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]
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
#methods = ["bg", "nm", "pw" ,"gd", "gdf", 
#"ncg", "tnc", "dog"]
#methods_cgf = ["cgf_fr", "cgf_pr", "cgf_prb"]
#methods = ["dog", "trn", "trk", "tre"]
methods_cgf = ["cgf_fr"]
methods = []
for pt in perts:
    #fig, ax = plt.subplots()
    #width = np.array(obs_s)*0.025
    #xaxis = np.array(obs_s) - 3.5*width
    j = 0
    i = 0
    emean = np.zeros((1+len(methods_cgf)+len(methods),len(obs_s)))
    dmean = 0.0
    score = open(f"score_{pt}_{op}.txt", "w")
    score.write(f"{pt} , operator : {op}, restart : False\n")
    score.write("method    type ")
    for err in obs_s:
        score.write("{:5.3e} ".format(err))
    score.write("all\n")
    score.write("===\n")
#    method = "lb"
#    if pt[0:4] == "etkf":
#        score.write(f"{pt[5:].ljust(7)}   mean ")
#    #elif pt == "mlefw" or pt == "gradw":
#    #    method = "ncg"
#    #    score.write(f"{method.ljust(7)}   mean ")
#    else:
#        score.write(f"{method.ljust(7)}   mean ")
#    for oberr in oberrs:
#        #f = "{}_e_{}_{}_oberr{}_mean.txt".format(model, op, pt, oberr)
#        if pt[0:4] == "etkf":
#            fm = "etkf_oberr/{}_{}_oberr{}_mean.txt".format(op, pt, oberr)
#        #elif pt == "mlefw" or pt == "gradw":
#        #    fm = "newton_oberr/{}_{}_oberr{}_ncg_mean.txt".format(op, pt, oberr)
#        else:
#            #f = "mlef_oberr/{}_{}_oberr{}_mean.txt".format(op, pt, oberr)
#            fm = "lb_rest2_oberr/{}_{}_oberr{}_lb_mean.txt".format(op, pt, oberr)
#            #f = "{}_e_{}_{}_oberr{}_{}.txt".format(model, op, pt, oberr, method)
#        #f = "{}_e_{}_{}_oberr{}.txt".format(model, op, pt, oberr)
#        if not os.path.isfile(fm):
#            print("not exist {}".format(fm))
#            score.write("nan".ljust(10))
#            emean[j,i] = np.nan
#            i += 1
#            continue
#        e = np.loadtxt(fm)
#        score.write("{:5.3e} ".format(np.mean(e[5:])))
#        emean[j,i] = np.mean(e[5:])
#        dmean += np.log10(np.mean(e[5:])/obs_s[oberrs.index(oberr)])
#        i += 1
#    dmean /= i
#    score.write("{:5.3e} ".format(dmean))
#    score.write("\n")
    j += 1
    #if pt[0:4] != "etkf":
    #if len(pt) < 5:
    for method in methods_cgf:
        score.write(f"{method.ljust(7)}   mean ")
        i = 0
        dmean = 0.0
        for oberr in oberrs:
            #fm = "cgf_rest_oberr/{}_{}_oberr{}_{}_mean.txt".format(op, pt, oberr, method)
            fm = "linear_oberr/{}_{}_oberr{}_{}_mean.txt".format(op, pt, oberr, method)
            if not os.path.isfile(fm):
                print("not exist {}".format(fm))
                score.write("nan".ljust(10))
                emean[j,i] = np.nan
                i += 1
                continue
            e = np.loadtxt(fm)
            score.write("{:5.3e} ".format(np.mean(e[5:])))
            emean[j,i] = np.mean(e[5:])
            dmean += np.log10(np.mean(e[5:])/obs_s[oberrs.index(oberr)])
            i += 1
        dmean /= i
        score.write("{:5.3e} ".format(dmean))
        score.write("\n")
        j += 1
#        for method in methods:
#            score.write(f"{method.ljust(7)}   mean ")
#            i = 0
#            dmean = 0
#            for oberr in oberrs:
#                #fm = "mlef-rest_oberr/{}_{}_oberr{}_{}_mean.txt".format(op, pt, oberr, method)
#                fm = "w-trust_oberr/{}_{}_oberr{}_{}_mean.txt".format(op, pt, oberr, method)
#                if not os.path.isfile(fm):
#                    print("not exist {}".format(fm))
#                    score.write("nan".ljust(10))
#                    emean[j,i] = np.nan
#                    i += 1
#                    continue
#                e = np.loadtxt(fm)
#                score.write("{:5.3e} ".format(np.mean(e[5:])))
#                emean[j,i] = np.mean(e[5:])
#                dmean += np.log10(np.mean(e[5:])/obs_s[oberrs.index(oberr)])
#                i += 1
#            dmean /= i
#            score.write("{:5.3e} ".format(dmean))
#            score.write("\n")
#            j += 1
#    score.write("\n")
#    method = "lb"
    j = 0
    i = 0
#    if pt[0:4] == "etkf":
#        score.write(f"{pt[5:].ljust(7)}   stdv ")
#    elif pt == "mlefw" or pt == "gradw":
#        method = "ncg"
#        score.write(f"{method.ljust(7)}   stdv ")
#    else:
#        score.write(f"{method.ljust(7)}   stdv ")
#    for oberr in oberrs:
#        #f = "{}_e_{}_{}_oberr{}_mean.txt".format(model, op, pt, oberr)
#        if pt[0:4] == "etkf":
#            fs = "etkf_oberr/{}_{}_oberr{}_stdv.txt".format(op, pt, oberr)
#        elif pt == "mlefw" or pt == "gradw":
#            fs = "newton_oberr/{}_{}_oberr{}_ncg_stdv.txt".format(op, pt, oberr, method)
#        else:
#            #f = "mlef_oberr/{}_{}_oberr{}_mean.txt".format(op, pt, oberr)
#            fs = "lb_rest2_oberr/{}_{}_oberr{}_lb_stdv.txt".format(op, pt, oberr)
#            #f = "{}_e_{}_{}_oberr{}_{}.txt".format(model, op, pt, oberr, method)
#        #f = "{}_e_{}_{}_oberr{}.txt".format(model, op, pt, oberr)
#        if not os.path.isfile(fs):
#            print("not exist {}".format(fs))
#            score.write("nan".ljust(10))
#            #el[i] = np.nan
#            i += 1
#            continue
#        e = np.loadtxt(fs)
#        score.write("{:5.3e} ".format(np.mean(e[5:])/emean[j,i]))
#        i += 1
#    score.write("\n")
    j += 1
#    if pt[0:4] != "etkf":
    #if len(pt) < 5:
    for method in methods_cgf:
        i = 0
        score.write(f"{method.ljust(7)}   stdv ")
        for oberr in oberrs:
            #fs = "cgf_rest_oberr/{}_{}_oberr{}_{}_stdv.txt".format(op, pt, oberr, method)
            fs = "linear_oberr/{}_{}_oberr{}_{}_stdv.txt".format(op, pt, oberr, method)
            if not os.path.isfile(fs):
                print("not exist {}".format(fs))
                score.write("nan".ljust(10))
                #el[i] = np.nan
                i += 1
                continue
            e = np.loadtxt(fs)
            score.write("{:5.3e} ".format(np.mean(e[5:])/emean[j,i]))
            i += 1
        score.write("\n")
        j += 1
#        for method in methods:
#            i = 0
#            score.write(f"{method.ljust(7)}   stdv ")
#            for oberr in oberrs:
#                #fs = "mlef-rest_oberr/{}_{}_oberr{}_{}_stdv.txt".format(op, pt, oberr, method)
#                fs = "w-trust_oberr/{}_{}_oberr{}_{}_stdv.txt".format(op, pt, oberr, method)
#                if not os.path.isfile(fs):
#                    print("not exist {}".format(fs))
#                    score.write("nan".ljust(10))
#                    #el[i] = np.nan
#                    i += 1
#                    continue
#                e = np.loadtxt(fs)
#                score.write("{:5.3e} ".format(np.mean(e[5:]/emean[j,i])))
#                i += 1
#            score.write("\n")
#            j += 1
    score.close()