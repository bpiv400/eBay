#!/bin/bash
#$ -N packages
#$ -j y
#$ -m e -M 4102158912@vtext.com

source /opt/rh/rh-python36/enable
source ~/.virtualenvs/jupyterlab-py36/bin/activate
pip install pandas
pip install bokeh
pip install nltk
pip install textblob
pip install spacy
pip install gensim
pip install numpy
pip install scipy
pip install sklearn
pip install matplotlib
pip install bokeh
pip install seaborn
pip install plotly
pip install theanos
pip install theano
pip install keras
pip install tensorflow
pip install statsmodels
