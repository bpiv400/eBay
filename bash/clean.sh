#!/bin/bash
#$ -l m_mem_free=75G
#$ -N clean
#$ -j y
#$ -o logs/

# stata do repo/clean/a_listings.do
# stata do repo/clean/b_clean.do
# stata do repo/clean/c_bins.do
# stata do repo/clean/d_conform.do
stata do repo/clean/e_flag.do
stata do repo/clean/f_csvs.do
