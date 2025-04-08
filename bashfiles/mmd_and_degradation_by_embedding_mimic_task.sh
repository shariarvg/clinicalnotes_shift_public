#!/bin/sh
export PYTHONPATH=../everything:$PYTHONPATH

arg1=$1

python3 ../pythonscripts/mmd_and_degradation_by_embedding_mimic_task.py $arg1