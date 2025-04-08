#!/bin/sh
export PYTHONPATH=../everything:$PYTHONPATH

arg1=$1
arg2=$2
arg3=$3
arg4=$4

python3 ../pythonscripts/mmd_and_degradation_by_embedding.py $arg1 $arg2 $arg3 $arg4