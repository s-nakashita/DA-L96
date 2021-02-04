#!/bin/sh
model="z08"
exp="dlag"
operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
#operators="cubic"
na="20"
if [ $# -gt 3 ]; then
  model=${1}
  op=${2}
  na=${3}
  src=${4}
fi
plots="e chi ua k cj trpa dy"
#for var in e dh du jh gh chi; do
for var in ${plots}; do
  echo ${op} ${var}
  cp ${src}/plot${var}.py plot.py
  python plot.py ${op} ${model} ${na}
  rm plot.py
done
