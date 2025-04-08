#!/bin/sh
export PYTHONPATH=../everything:$PYTHONPATH

arg1=$1

python3 ../pythonscripts/degradation_script.py $arg1