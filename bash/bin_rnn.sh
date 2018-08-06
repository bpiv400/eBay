#$ -N bin 
#!/bin/bash
#$ -js 1
#$ -j y
#$ -l m_mem_free=100G

while getopts 'l:h:s:n:t:p:d:ae:br:' flag; do
  case "${flag}" in
    l) low="${OPTARG}" ;;
    h) high="${OPTARG}" ;;
    s) step="${OPTARG}" ;;
    p) perc="${OPTARG}" ;;
    d) tol="${OPTARG}" ;;
    a) abs="True" ;;
    e) exp="${OPTARG}" ;;
    b) bin="True" ;;
    r) roundLevel="${OPTARG}" ;;
  esac
done
echo $turn
echo $name
echo $abs
cd ~/eBay/data/$name

if [ -n "$bin" ]; then
    scriptPath=repo/rnn/bin_rnn.py ;
else
    scriptPath=repo/rnn/bin_rnn_dnorm.py ;
fi

cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
types=( "toy" "train" "test" )
for k in "${types[@]}"
do
    echo $k
    if [ -z ${step+x} ]; then
        if [ -z "$abs" ]; then
            if [ -n "$perc" ]; then
                echo "Common, not stepwise"
                python "$scriptPath" --name $k --low $low --exp $exp --high $high --turn $j --num $perc  --tol $tol ;
            else
        echo "Using normalized difference"
        python $scriptPath --name $k --exp $exp --sig $roundLevel ;
    fi
        else
            echo "Using absolute difference"
            python "$scriptPath" --name $k --low $low --exp $exp --high $high --turn $j --num $perc  --abs --tol $tol ;
        fi
    else
        python "$scriptPath" --name $k --exp $exp --step $step --low $low --high $high --turn $j ;
    fi
done
