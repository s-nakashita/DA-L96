#!/bin/sh
#set -x
#operators="linear test"
operators="quadratic cubic quadratic-nodiff cubic-nodiff" # quartic quartic-nodiff"
#perturbations="etkf-jh etkf-fh mlef grad" # po srf letkf"
#perturbations="mlef05"
perturbations="mlef grad mlefb mleft mlef05 grad05 mlefw" # mlef3 mlefh"
na=20
linf="F"
lloc="F"
ltlm="F"
model=z08
#vname="oberr"
exp="minimize_method"
echo ${exp} ${vname}
#sigma="0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001"
#sigma="0.1 0.01 0.001 0.0001"
#lags="4 6 8 10 12 14 16 18"
methods="lb bg cg nm gd"
#rm z08*.txt
#rm z08*.npy
#rm z08*.log
src=$(pwd)
rm -rf ${exp}
mkdir ${exp}
cd ${exp}
#cp ${src}/iobs.py .
#cp ${src}/calc_mean.py .
cp ${src}/output.sh .
cp ${src}/copy.sh .
cp ${src}/plot.sh .
cp ${src}/logging_config.ini .
for op in ${operators}; do
  #./clean.sh  z08 ${op}
#for obs_s in $sigma ; do
  #obs_s=0.01
#for lag in $lags ; do
for method in $methods ; do
  #var=
  #var=${lag} 
  #var=${obs_s}
  var=${method}
  #ivar=0
  #ivar=${lag} 
  #ivar=$(python iobs.py ${obs_s})
  ivar=${var}
  #exp=${vname}${ivar}
  echo ${exp} ${vname}
  for pert in ${perturbations}; do
    pt=${pert}
    #pt=${pert:0:4}
    if test "${pert:5:2}" = "jh" ; then
      ltlm="T"
    elif test "${pert:5:2}" = "fh" ; then
      ltlm="F"
    fi
    #for count in $(seq 1 50); do
    echo ${op} ${pt} ${linf} ${lloc} ${ltlm} ${var}
    echo ${vname} ${na} ${ivar}
    python ${src}/z08.py ${op} ${pt} ${linf} ${lloc} ${ltlm} ${var} > z08_${op}_${pert}.log 2>&1
    wait
    #tail -1 z08_e_${op}_${pt}.txt
    #tail -1 z08_e_${op}_${pt}_${vname}${ivar}.txt
    tail -1 z08_e_${op}_${pt}_${var}.txt
    #cp z08_e_${op}_${pt}_${vname}${ivar}.txt z08_e_${op}_${pert}_${vname}${ivar}.txt
    #mv z08_e_${op}_${pt}.txt z08_e_${op}_${pert}.txt
    #cp z08_chi_${op}_${pt}_${vname}${ivar}.txt z08_chi_${op}_${pt}.txt
    #mv z08_${op}_${pert}.log z08_${op}_${pert}_${vname}${ivar}.log
    #mv z08_K2_${op}_${pt}_cycle0.npy z08_K2_${op}_${pert}_cycle0_${vname}${ivar}.npy 
    #mv z08_K_${op}_${pt}_cycle0.npy z08_K_${op}_${pert}_cycle0_${exp}.npy 
    #mv z08_Kloc_${op}_${pt}_cycle0.npy z08_Kloc_${op}_${pert}_cycle0_${exp}.npy 
    #mv z08_e_${op}_${pt}.txt e${ivar}_${count}.txt
    #mv z08_chi_${op}_${pt}.txt chi${ivar}_${count}.txt
    #mv z08_e_${op}_${pt}_${vname}${ivar}.txt ${vname}${ivar}_${count}.txt
    #rm obs*.npy
    #done # for count
    #python calc_mean.py ${vname} ${na} ${ivar} ${count}
    #mv ${vname}${ivar}_mean.txt z08_e_${op}_${pt}_${vname}${ivar}_mean.txt
    #rm ${vname}${ivar}*.txt
    #python calc_mean.py e ${na} ${ivar} ${count}
    #mv e${ivar}_mean.txt z08_e_${op}_${pt}.txt
    #rm e${ivar}_*.txt
    #python calc_mean.py chi ${na} ${ivar} ${count}
    #mv chi${ivar}_mean.txt z08_chi_${op}_${pt}.txt
    #rm chi${ivar}_*.txt
    #./output.sh ${exp} z08 ${op} ${pt} ${pert}
    #python ../plotcj2d.py ${op} z08 ${na} ${pert}
  done # for perturbation
  rm obs*.npy
  #python plotcJb+o.py ${op} z08 ${na}
  #for i in $(seq 0 3); do
  #  mv z08_cJb_${op}_cycle${i}.png z08_cJb_${op}_cycle${i}_${exp}${ivar}.png
  #  mv z08_cJo_${op}_cycle${i}.png z08_cJo_${op}_cycle${i}_${exp}${ivar}.png
  #done
  #./copy.sh z08 ${exp} ${op}
  #./plot.sh z08 ${op} ${na} ${src}
  #for pt in ${perturbations}; do
  #  convert -delay 10 z08_ua_${op}_${pt}_cycle*.png z08_ua_${op}_${pt}.gif
  #done
  #python ${src}/plote.py ${op} z08 ${na}
  #python ${src}/plotcg.py ${op} z08 ${na}
#./copy.sh z08 ${exp} ${op}
#python plotcj.py ${op} z08 ${na}
#for i in $(seq 0 3); do
#  pdfcrop z08_cJ_${op}_cycle${i}.pdf z08_cJ_${op}_cycle${i}_${exp}.pdf
#done
#plot=e-lag
#python plotcondk2.py ${op} z08 ${na}
#python plotsmw.py ${op} z08 ${na} ${obs_s}
#python plotpf.py ${op} z08 ${na} ${obs_s}
#python plotdhdx.py ${op} z08 ${na} ${obs_s}
#python plotdy.py ${op} z08 ${na} ${obs_s}
#python plotdy.py ${op} z08 ${na}
#mv ${model}_dy_${op}_etkf.png ${model}_dy_${op}_etkf_${exp}.png
#mv ${model}_dy_${op}_mlef.png ${model}_dy_${op}_mlef_${exp}.png
#python plote.py ${op} ${model} ${na}
#mv ${model}_e_${op}.png ${model}_e_${op}_${exp}.png
#rm ${model}_e_*.txt
#pdfcrop ${model}_e_${op}.pdf ${model}_e_${op}_${exp}.pdf
#python plottrpa.py ${op} ${model} ${na}
#mv ${model}_trpa_${op}.png ${model}_trpa_${op}_${exp}.png
#pdfcrop ${model}_trpa_${op}.pdf ${model}_trpa_${op}_${exp}.pdf
#python plotua.py ${op} ${model} ${na}
#python plotdh.py ${op} ${model} ${na}
#python plotk.py ${op} ${model} ${na}
#python plotk-etkf.py ${op} ${model} ${na}
#mv ${model}_k_${op}_etkf.png ${model}_k_${op}_etkf_${exp}.png
#python plotdof.py ${op} ${model} ${na}
#mv ${model}_dof_${op}.png ${model}_dof_${op}_${exp}.png
#for pt in ${perturbations}; do
#  mv ${model}_ua_${op}_${pt}.png ${model}_ua_${op}_${pt}_${exp}.png
#  mv ${model}_dh_${op}_${pt}.png ${model}_dh_${op}_${pt}_${exp}.png
#  mv ${model}_kb_${op}_${pt}.png ${model}_kb_${op}_${pt}_${exp}.png
#  mv ${model}_kl_${op}_${pt}.png ${model}_kl_${op}_${pt}_${exp}.png
#done
#python plottrpf.py ${op} ${model} ${na}
#mv ${model}_trpf_${op}.png ${model}_trpf_${op}_${exp}.png
done # for obs_s
for pt in ${perturbations}; do
  python ../plotemethod.py ${op} z08 ${na} ${pt}
done
#plot=eoberr
#cp ${src}/plot${plot}.py .
#python plot${plot}.py ${op} z08 ${na}
#pdfcrop z08_${plot}_${op}+nodiff.pdf z08_${plot}_${op}+nodiff_${exp}.pdf
done # for operator
#for op in cubic; do
##  plot=eoberr
##  python plot${plot}.py ${op} z08 ${na}
##  pdfcrop z08_${plot}_${op}+nodiff.pdf z08_${plot}_${op}+nodiff_${exp}.pdf#  echo ${op} e
#  python plote.py ${op} ${model} ${na}
#  mv ${model}_e_${op}+nodiff.png ${model}_e_${op}+nodiff_${exp}.png
#  pdfcrop ${model}_e_${op}+nodiff.pdf ${model}_e_${op}+nodiff_${exp}.pdf
#done
rm obs*.npy
mv z08*.txt ${src}/numeric/z08/
mv z08*.npy ${src}/numeric/z08/