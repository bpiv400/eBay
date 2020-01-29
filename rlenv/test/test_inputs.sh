#$ -l m_mem_free=128G
#$ -N test_inputs
#$ -j y
#$ -o logs/
#$ -q all.q

python repo/rlenv/test/test_inputs.py --part "$1" --num "$2"
