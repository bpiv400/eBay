#$ -N bin 
#!/bin/bash
#$ -js 1
#$ -j y
#$ -l m_mem_free=25G

while getopts 'l:h:s:n:t:p:d:ae:n:t:' flag; do
  case "${flag}" in
    l) low="${OPTARG}" ;;
    h) high="${OPTARG}" ;;
    s) step="${OPTARG}" ;;
    p) perc="${OPTARG}" ;;
    d) tol="${OPTARG}" ;;
    a) abs="True" ;;
    e) exp="${OPTARG}" ;;
    n) name="${OPTARG}" ;;
    t) turn="${OPTARG}" ;;
  esac
done
echo $turn
echo $name
echo $abs
cd ~/eBay/data/$name
scriptPath=repo/trans_probs/mvp/bin.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
if [ -z ${step+x} ]; then
    echo "Common, not stepwise"
    if [-z ${abs+x} ]; then
          echo "Using percent difference instead of absolute difference"
          python "$scriptPath" --name $name --low $low --exp $exp --high $high --turn $turn --num $perc  --tol $tol ;
    else
          echo "Using absolute difference"
          python "$scriptPath" --name $name --low $low --exp $exp --high $high --turn $turn --num $perc  --abs --tol $tol ;
    fi
else
    python "$scriptPath" --name $name --exp $exp --step $step --low $low --high $high --turn $turn ;
fi
