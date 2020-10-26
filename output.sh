#!/bin/sh
if [ $# -lt 4 ]; then
    echo "Usage :: $0 exp model operator datype [lag]"
    exit
fi
exp=${1}
model=${2} #z08,l96
op=${3}    #linear,quadratic,cubic,quadratic-nodiff,cubic-nodiff
pt=${4}    #mlef,grad,etkf,po,srf,letkf
lag=0
if [ $# -gt 4 ]; then
    lag=${5}
fi
# output file name example
# z08_uf_linear_mlef.npy z08_ua_linear_mlef.npy z08_pa_linear_mlef.npy
# z08_dh_linear_mlef_cycle0.npy z08_cJ_linear_mlef_cycle0.npy
# z08_e_linear_mlef.txt(z08_e_linear_mlef_lag8.txt) z08_chi_linear_mlef.txt 
# z08_jh_linear_mlef_cycle0.txt z08_gh_linear_mlef_cycle0.txt
npy_header="MOD_uf MOD_ua MOD_pa"
npy_headerc="MOD_dh MOD_cJ"
txt_header="MOD_e MOD_chi"
txt_headerc="MOD_jh MOD_gh"

for f in ${npy_header};do
    f=${f/MOD/$model}
    inf=${f}_${op}_${pt}.npy
    echo $inf
    mv ${inf} ${inf%.npy}_${exp}.npy
done
for f in ${npy_headerc};do
    for i in $(seq 0 3);do
        f=${f/MOD/$model}
        inf=${f}_${op}_${pt}_cycle${i}.npy
        echo $inf
        mv ${inf} ${inf%.npy}_${exp}.npy
    done
done
for f in ${txt_header};do
    f=${f/MOD/$model}
    inf=${f}_${op}_${pt}.txt
    echo $inf
    mv ${inf} ${inf%.txt}_${exp}.txt
done
for f in ${txt_headerc};do
    for i in $(seq 0 3);do
        f=${f/MOD/$model}
        inf=${f}_${op}_${pt}_cycle${i}.txt
        echo $inf
        mv ${inf} ${inf%.txt}_${exp}.txt
    done
done

ls -ltr | tail -n 9