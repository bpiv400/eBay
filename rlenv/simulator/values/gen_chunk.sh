#$ -l m_mem_free=4G
#$ -N value_generator
#$ -j y
#$ -o logs/
#$ -q short.q

python repo/rlenv/simulator/generate.py --values --part "$1" --num "$2"
