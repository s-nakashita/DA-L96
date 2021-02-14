#!/bin/sh
if [ $# -lt 2 ]; then
    echo "Usage :: $0 model operator"
    exit
fi
model=${1} #z08,l96
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
op=${2}
perturbations="mlef grad etkf po srf letkf"
# output file name example
# z08_uf_linear_mlef.npy z08_ua_linear_mlef.npy z08_pa_linear_mlef.npy
# z08_dh_linear_mlef_cycle0.npy z08_cJ_linear_mlef_cycle0.npy
# z08_e_linear_mlef.txt(z08_e_linear_mlef_lag8.txt) z08_chi_linear_mlef.txt 
# z08_jh_linear_mlef_cycle0.txt z08_gh_linear_mlef_cycle0.txt
npy_header="MOD_uf MOD_ua MOD_pa"
npy_headerc="MOD_dh MOD_cJ"
txt_header="MOD_e MOD_chi"
txt_headerc="MOD_jh MOD_gh"
for pt in ${perturbations};do
for f in ${npy_header};do
    f=${f/MOD/$model}
    inf=${f}_${op}_${pt}.npy
    rm $inf
done
for f in ${npy_headerc};do
    for i in $(seq 0 3);do
        f=${f/MOD/$model}
        inf=${f}_${op}_${pt}_cycle${i}.npy
        rm $inf
    done
done
for f in ${txt_header};do
    f=${f/MOD/$model}
    inf=${f}_${op}_${pt}.txt
    rm $inf
done
for f in ${txt_headerc};do
    for i in $(seq 0 3);do
        f=${f/MOD/$model}
        inf=${f}_${op}_${pt}_cycle${i}.txt
        rm $inf
    done
done
rm ${model}_${op}_${pt}.log
done

ls -ltr | tail -n 9