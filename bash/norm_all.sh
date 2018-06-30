#!/bin/bash

qsub -js 1 repo/bash/mvp_norm.sh -e mvp1 
qsub -js 1 repo/bash/mvp_norm.sh -e mvp2
qsub -js 1 repo/bash/mvp_norm.sh -e mvp3
qsub -js 1 repo/bash/mvp_norm.sh -e mvp4
