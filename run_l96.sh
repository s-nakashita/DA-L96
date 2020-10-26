#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear"
#perturbations="mlef grad etkf po srf letkf"
perturbations="mlef grad" # etkf letkf"
na=200 # Number of assimilation cycle
linf="T"
lloc="F"
ltlm="F"
exp="nmem"
echo ${exp}
./clean.sh  l96 ${operators}
for pt in ${perturbations}; do
  for op in ${operators}; do
    echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm}
    python l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} > l96_${op}_${pt}.log 2>&1
    wait
    ./output.sh ${exp} l96 ${op} ${pt}
  done
done
./plot.sh l96 ${exp} ${operators} ${na}
#./copy.sh l96 ${exp} ${operators}
#python plote.py ${operators} l96 ${na}
#mv l96_e_${operators}.png l96_e_${operators}_${exp}.png