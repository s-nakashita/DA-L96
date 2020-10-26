#!/bin/sh
model="z08"
exp="nmem"
operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
#operators="cubic"
na="20"
if [ $# -gt 3 ]; then
  model=${1}
  exp=${2}
  operators=${3}
  na=${4}
fi
for op in ${operators}; do
./copy.sh ${model} ${exp} ${op}
done
#for var in e dh du jh gh chi; do
for var in e chi; do
  for op in ${operators}; do
    echo ${op} ${var}
    python plot${var}.py ${op} ${model} ${na}
  done
  mv ${model}_${var}_${op}.png ${model}_${var}_${op}_${exp}.png
done
for var in jh gh cJ; do
  for op in ${operators}; do
    echo ${op} ${var}
    python plot${var}.py ${op} ${model} ${na}
  done
  for i in $(seq 0 3); do
    mv ${model}_${var}_${op}_cycle${i}.png ${model}_${var}_${op}_cycle${i}_${exp}.png
  done
done
#for var in pa pat dh; do
#  for op in ${operators}; do
#    echo ${op} ${var}
#    python plot${var}.py ${op} ${model} ${na}
#  done
#  for pt in mlef grad etkf po srf letkf;do
#    mv ${model}_${var}_${op}_${pt}.png ${model}_${var}_${op}_${pt}_${exp}.png
#  done
#done