#!/bin/sh
export PYTHONPATH=../everything:$PYTHONPATH

arg1=$1
arg2=$2

#python3 ../pythonscripts/fine_tune_classifier.py
python3 ../pythonscripts/tabular_fitting_script.py $arg1 $arg2