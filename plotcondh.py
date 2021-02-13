import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["etkf", "po", "srf", "letkf"]
perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
if model == "l96" and op == "linear":
    perts = ["mlef", "etkf", "po", "srf", "letkf"]
    #na = 100
elif model == "z08":
    perts = ["mlef", "grad", "mlefb", "mleft", "etkf", "etkf-fh", "etkf-jh"]#, "po", "srf", "letkf"]
    linestyle = {"mlef":"solid", "grad":"dashed", "mlefb":"solid", "mleft":"solid",
     "etkf":"solid", "etkf-fh":"solid", "etkf-jh":"dashed"}
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange', "mlefb":"tab:cyan", "mleft":"tab:pink", "etkf":'tab:green', "etkf-fh":'tab:green', "etkf-jh":'tab:red'}
x = np.arange(na) + 1
y = np.ones(x.shape)
fcond = dict()
ax = plt.subplot(221)
for pt in perts:
    f = "{}_condh_{}_{}.txt".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    cond = np.loadtxt(f)
    #if pt == "mlef" or pt == "mlefb":
    #    ax.plot(x[1:], cond[1:,0], linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    #    fcond[pt] = cond[0,0]
    #else:
    ax.plot(x[1:], cond[1:], linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    fcond[pt] = cond[0]
#ax.plot(x, y, linestyle="dotted", color='tab:purple')
#ax.set_yscale("log")
#if np.max(chi) > 1000.0:
#    ax.set_ylim(0.1, 1000.0)
#    ax.set_yticks([1,10,100,1000])
#if np.max(chi) > 10000.0:
#    ax.set_ylim(0.1, 10000.0)
#    ax.set_yticks([1,10,100,1000,10000])
ax.set(xlabel="analysis cycle", ylabel="cond(Hessian)",
        title=op)
ax.set_xticks(x[::len(x)//10])
ax.set_xticks(x[::len(x)//20], minor=True)
ax.legend()
plt.subplot(212)
plt.text(0, 0.2, "First condition number\nmlef :{:8.3f}".format(fcond["mlef"])+"\nmlefb:{:8.3f}".format(fcond["mlefb"])+"\nmleft:{:8.3f}".format(fcond["mleft"])+"\netkf :{:8.3f}".format(fcond["etkf"]), size=14,
         va="baseline", ha="left", multialignment="left",
         bbox=dict(fc="none"))
plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
plt.savefig("{}_condh_{}.png".format(model, op))
plt.close()
#plt.show()
#fconde = dict()
#ax = plt.subplot(221)
#for pt in perts:
#    f = "{}_condh_{}_{}.txt".format(model, op, pt)
#    if not os.path.isfile(f):
#        print("not exist {}".format(f))
#        continue
#    cond = np.loadtxt(f)
#    if pt == "mlef" or pt == "mlefb":
#        ax.plot(x[1:], cond[1:,1], linestyle=linestyle[pt], color=linecolor[pt], label=pt+"(estimate)")
#        fconde[pt] = cond[0,1]
#ax.set(xlabel="analysis cycle", ylabel="cond(Hessian)",
#        title=op)
#ax.set_xticks(x[::len(x)//10])
#ax.set_xticks(x[::len(x)//20], minor=True)
#ax.legend()
#plt.subplot(212)
#plt.text(0, 0.2, "First condition number (estimate)\nmlef :{:8.3f}".format(fconde["mlef"])+"\nmlefb:{:8.3f}".format(fconde["mlefb"]), size=14,
#         va="baseline", ha="left", multialignment="left",
#         bbox=dict(fc="none"))
#plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
#plt.savefig("{}_condhe_{}.png".format(model, op))
#plt.close()