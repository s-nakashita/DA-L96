#!/bin/sh
operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
#operators="linear"
perturbations="mlef grad etkf po srf letkf"
#perturbations="mlef grad"
for pt in ${perturbations}; do
  for op in ${operators}; do
    echo ${op} ${pt}
    python z08.py ${op} ${pt} > z08_${op}_${pt}.log 2>&1
    #if [ ${pt} = "mlef" ] || [ ${pt} = "grad" ]; then
    #  python costJ.py ${op} ${pt} > ${op}_${pt}.log 2>&1
    #fi
  done
done
