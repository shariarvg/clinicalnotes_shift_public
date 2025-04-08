#!/bin/sh
export PYTHONPATH=../everything:$PYTHONPATH

arg1=$1

python3 ../pythonscripts/drift_mortality_script.py $arg1