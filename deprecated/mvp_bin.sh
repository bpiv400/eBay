#$ -N bin 
#!/bin/bash
#$ -js 1
#$ -j y
#$ -l m_mem_free=25G

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
    scriptPath=repo/trans_probs/mvp/bin.py ;
else
    scriptPath=repo/trans_probs/mvp/bin_dnorm.py ;
fi

cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
types=( "toy" "train" "test" )
turns=( "b0" "b1" "b2" )
for k in "${types[@]}"
do
    for j in "${turns[@]}" 
    do
        if [ -z ${step+x} ]; then
            if [ -z "$abs" ]; then
                if [ -n "$perc" ]; then 
            	    echo "Common, not stepwise"
                    python "$scriptPath" --name $k --low $low --exp $exp --high $high --turn $j --num $perc  --tol $tol ;
                else
 		    echo "Using normalized difference"
 		    python $scriptPath --name $k --exp $exp --turn $j --sig $roundLevel ;
		        fi
            else
                echo "Using absolute difference"
                python "$scriptPath" --name $k --low $low --exp $exp --high $high --turn $j --num $perc  --abs --tol $tol ;
            fi
        else
            python "$scriptPath" --name $k --exp $exp --step $step --low $low --high $high --turn $j ;
        fi
    done
done
