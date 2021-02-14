#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="quadratic cubic quadratic-nodiff cubic-nodiff"
#perturbations="etkf-jh etkf-fh mlef grad" # po srf letkf"
perturbations="mlef grad"
#perturbations="po etkf letkf"
na=20
linf="F"
lloc="F"
ltlm="T"
exp="nrm"
vname="nrm"
echo ${exp} ${vname}
sigma="0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001"
lags="4 6 8 10 12 14 16 18"
rm z08*.txt
rm z08*.npy
rm z08*.log
for op in ${operators}; do
  ivar=0
  for pert in ${perturbations}; do
    pt=${pert:0:4}
    if test "${pert:5:2}" = "jh" ; then
      ltlm="T"
    elif test "${pert:5:2}" = "fh" ; then
      ltlm="F"
    fi
    for count in $(seq 1 50); do
    echo ${op} ${pt} ${linf} ${lloc} ${ltlm} 
    echo ${vname} ${na} ${ivar}
    python z08-2.py ${op} ${pt} ${linf} ${lloc} ${ltlm} > z08_${op}_${pt}.log 2>&1
    wait
    tail -1 z08_e_${op}_${pt}.txt
    #tail -1 z08_e_${op}_${pt}_${vname}${ivar}.txt
    mv z08_${op}_${pt}.log z08_${op}_${pert}.log
    #mv z08_K_${op}_${pt}_cycle0.npy z08_K_${op}_${pert}_cycle0_${exp}.npy 
    #mv z08_Kloc_${op}_${pt}_cycle0.npy z08_Kloc_${op}_${pert}_cycle0_${exp}.npy 
    mv z08_e_${op}_${pt}.txt e${ivar}_${count}.txt
    mv z08_chi_${op}_${pt}.txt chi${ivar}_${count}.txt
    #mv z08_e_${op}_${pt}_${vname}${ivar}.txt ${vname}${ivar}_${count}.txt
    #rm obs*.npy
    done
    python calc_mean.py e ${na} ${ivar} ${count}
    mv ${vname}${ivar}_mean.txt z08_e_${op}_${pt}_${vname}${ivar}_mean.txt
    mv e${ivar}_mean.txt z08_e_${op}_${pt}.txt
    rm e${ivar}_*.txt
    python calc_mean.py chi ${na} ${ivar} ${count}
    mv ${vname}${ivar}_mean.txt z08_e_${op}_${pt}_${vname}${ivar}_mean.txt
    mv chi${ivar}_mean.txt z08_chi_${op}_${pt}.txt
    rm chi${ivar}_*.txt
    ./output.sh ${exp} z08 ${op} ${pt} ${pert}
  done
./plot.sh z08 ${exp} ${op} ${na}
done
#rm obs*.npy
#mv z08*.txt numeric/z08/
#mv z08*.npy numeric/z08/