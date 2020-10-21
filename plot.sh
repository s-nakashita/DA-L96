#!/bin/sh
model="z08"
operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
#operators="linear"
for var in e dh du jh gh cJ chi; do
  for op in ${operators}; do
    echo ${op} ${var}
    python plot${var}.py ${op} ${model}
  done
done
