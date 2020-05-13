#!/bin/bash
#
#  Execute from the current working directory
#$ -cwd
#
#  This is a long-running job
#$ -l day
#
#  Can use up to 6GB of memory
#$ -l vf=64
#
#$ -pe smp 4
#
#$ -t 1-8
#

source /data/nlp/ndo3/recursionrsaenv/bin/activate && python3.7 main.py
