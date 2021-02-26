#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear quadratic cubic"
perturbations="mlef etkf po srf letkf"
#perturbations="mlef mlefb mleft"
#perturbations="mlef grad etkf-jh etkf-fh"
na=100 # Number of assimilation cycle
linf="T"
lloc="F"
ltlm="F"
exp="l96_test"
echo ${exp}
src=$(pwd)
rm -rf ${exp}
mkdir -p ${exp}
cd ${exp}
cp ../data.csv .
cp ../logging_config.ini .
gamma="1 2 3 4 5 6 7 8 9 10"
inf="1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9"
for op in ${operators}; do
  #for ga in ${gamma}; do
    for pt in ${perturbations}; do
      #for infl_parm in ${inf}; do
      infl_parm=1.2
        #pt=${pert:0:4}
        pert=${pt}
        if test "${pert:5:2}" = "jh" ; then
          ltlm="T"
        elif test "${pert:5:2}" = "fh" ; then
          ltlm="F"
        fi
        #echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${ga}
        echo ${op} ${pt} ${na} ${infl_parm} ${lloc} ${ltlm} ${ga}
        #for count in $(seq 1 50); do
        #  echo ${count}
          #python ../l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${ga} > l96_${op}_${pert}.log 2>&1
          python ../l96.py ${op} ${pt} ${na} ${infl_parm} ${lloc} ${ltlm} > l96_${op}_${pert}.log 2>&1
          wait
          #mv l96_e_${op}_${pt}_ga${ga}.txt e${ga}_${count}.txt
          #rm obs*.npy
        #done
        #python ../calc_mean.py e ${na} ${ga} ${count}
        #if test ${pt} = "grad" ; then
        #  #mv e${ga}_mean.txt l96_e_${op}_mlef_ga${ga}_mean.txt
        #  mv e${ga}_mean.txt l96_e_${op}_mlef_${infl_parm}.txt
        #else
        #  #mv e${ga}_mean.txt l96_e_${op}_${pt}_ga${ga}_mean.txt
        #  mv e${ga}_mean.txt l96_e_${op}_${pt}_${infl_parm}.txt
        #fi
        #rm e${ga}_*.txt
      #done
    #./output.sh ${exp} l96 ${op} ${pt} ${pert}
    #python ../ploteparam.py ${op} l96 ${na} infl
    #if test ${pt} = "grad" ; then
    #  mv l96_einfl_${op}_mlef.png l96_einfl_${op}_${pt}_ga${ga}.png 
    #else
    #  mv l96_einfl_${op}_${pt}.png l96_einfl_${op}_${pt}_ga${ga}.png 
    #fi
    done
    #./copy.sh l96 ${exp} ${op}
    #python plote.py ${op} l96 ${na}
    #mv l96_e_${op}.png l96_e_${op}_${exp}.png
    #python plotdy.py ${op} l96 ${na}
    #python plotlpf.py ${op} l96 ${na}
    #python ../ploteparam.py ${op} l96 ${na} infl
  #done
  #python ../plotega.py ${op} l96 ${na} 
  python ../plote.py ${op} l96 ${na}
  python ../plotgh.py ${op} l96 ${na}
  python ../plotjh.py ${op} l96 ${na}
done
#./plot.sh l96 ${exp} ${operators} ${na}
#./copy.sh l96 ${exp} ${operators}
#python plote.py ${operators} l96 ${na}
#mv l96_e_${operators}.png l96_e_${operators}_${exp}.png
mv l96*.txt ../numeric/l96/
mv l96*.npy ../numeric/l96/