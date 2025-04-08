#! /bin/bash
export PYTHONPATH=../everything:$PYTHONPATH

arg1=$1

python3 ../pythonscripts/source_pairs_degradation_script.py $arg1