#!/bin/bash
#$ -V
#$ -cwd
#$ -j y                      # combine the stdout and stderr streams
#$ -N PrefixSUM              # specify the executable
#$ -l h_rt=1:30:00           # run time not to exceed one hour and 30 minutes
#$ -o $JOB_NAME.out$JOB_ID   # specify stdout & stderr output
#$ -q gpu                    # specify the GPU queue
#$ -pe 1way 12               # request one node (the 12)
module load cuda
set -x
./PrefixSUM
