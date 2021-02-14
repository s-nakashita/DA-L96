#!/bin/sh
if [ $# -lt 4 ]; then
    echo "Usage :: $0 exp model operator datype [lag]"
    exit
fi
exp=${1}
model=${2} #z08,l96
op=${3}    #linear,quadratic,cubic,quadratic-nodiff,cubic-nodiff
pt=${4}    #mlef,grad,etkf,po,srf,letkf
pert=${5}  #mlef,grad,etkf-jh,etkf-fh
lag=0
if [ $# -gt 5 ]; then
    lag=${6}
fi
# output file name example
# z08_uf_linear_mlef.npy z08_ua_linear_mlef.npy z08_pa_linear_mlef.npy
# z08_dh_linear_mlef_cycle0.npy z08_cJ_linear_mlef_cycle0.npy
# z08_e_linear_mlef.txt(z08_e_linear_mlef_lag8.txt) z08_chi_linear_mlef.txt 
# z08_jh_linear_mlef_cycle0.txt z08_gh_linear_mlef_cycle0.txt
npy_header="MOD_uf MOD_ua"
npy_headerc="MOD_dxf MOD_dhdx MOD_dy MOD_d MOD_cJ MOD_pf MOD_pa MOD_ua MOD_K MOD_K1 MOD_K2 MOD_K2i MOD_Kloc MOD_dx"
txt_header="MOD_e MOD_chi MOD_dof MOD_dpa MOD_ndpa"
txt_headerc="MOD_jh MOD_gh"

for f in ${npy_header};do
    f=${f/MOD/$model}
    inf=${f}_${op}_${pt}.npy
    #echo $inf
    mv ${inf} ${f}_${op}_${pert}_${exp}.npy
done
for f in ${npy_headerc};do
    for i in $(seq 0 3);do
        f=${f/MOD/$model}
        inf=${f}_${op}_${pt}_cycle${i}.npy
        #echo $inf
        mv ${inf} ${f}_${op}_${pert}_cycle${i}_${exp}.npy
    done
done
for f in ${txt_header};do
    f=${f/MOD/$model}
    inf=${f}_${op}_${pt}.txt
    #echo $inf
    mv ${inf} ${f}_${op}_${pert}_${exp}.txt
done
for f in ${txt_headerc};do
    for i in $(seq 0 9);do
        f=${f/MOD/$model}
        inf=${f}_${op}_${pt}_cycle${i}.txt
        #echo $inf
        mv ${inf} ${f}_${op}_${pert}_cycle${i}_${exp}.txt
    done
done

#ls -ltr | tail -n 9