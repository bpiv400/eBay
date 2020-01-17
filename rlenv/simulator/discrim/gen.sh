#$ -t 1-1000
#$ -l m_mem_free=6G
#$ -N chunk
#$ -j y
#$ -o logs/

python repo/rlenv/simulator/generate.py --part "$1" --num "$SGE_TASK_ID"