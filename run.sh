#!/bin/sh
operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
#operators="cubic-nodiff"
#perturbations="mlef grad etkf po srf letkf"
perturbations="mlef grad"
#perturbations="po etkf letkf"
linf="F"
lloc="F"
ltlm="T"
exp="nmem"
echo ${exp}
./clean.sh  z08
for pt in ${perturbations}; do
  for op in ${operators}; do
    #for obs_s in 0.1 0.01 0.001 0.0001 ; do
    echo ${op} ${pt} ${linf} ${lloc} ${ltlm} #${obs_s}
    python z08.py ${op} ${pt} ${linf} ${lloc} ${ltlm} > z08_${op}_${pt}.log 2>&1
    wait
    tail -1 z08_e_${op}_${pt}.txt
    ./output.sh ${exp} z08 ${op} ${pt}
    #done
  done
done
./plot.sh
