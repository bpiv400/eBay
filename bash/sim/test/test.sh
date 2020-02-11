#$ -l m_mem_free=4G
#$ -t 1-1000
#$ -N test_inputs
#$ -j y
#$ -o logs/test/
#$ -q short.q

python repo/rlenv/test/test_inputs.py --num "$SGE_TASK_ID" --part test
