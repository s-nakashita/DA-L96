import sys

pt = sys.argv[1]
gamma = int(sys.argv[2])
mlef = {1:1.2, 2:1.3, 3:1.2, 4:1.3, 5:1.4, 6:1.3, 7:1.3, 8:1.3, 9:1.3, 10:1.4}
etkf = {1:1.3, 2:1.2, 3:1.4, 4:1.4, 5:1.4, 6:1.3, 7:1.3, 8:1.3, 9:1.4, 10:1.4}
if pt == "mlef":
    print(mlef[gamma])
elif pt == "etkf":
    print(etkf[gamma])