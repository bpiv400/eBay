#!/bin/bash

qsub -js 1 repo/bash/mvp_bin.sh -l 3 -h 1000 -e mvp1 -s .5
qsub -js 1 repo/bash/mvp_bin.sh -l 3 -h 1000 -e mvp2 -s .5
qsub -js 1 repo/bash/mvp_bin.sh -l 3 -h 1000 -e mvp3 -a -d .5 -p .01
qsub -js 1 repo/bash/mvp_bin.sh -l 3 -h 1000 -e mvp4 -a -d .5 -p .01
