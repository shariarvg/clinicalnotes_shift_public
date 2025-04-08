#!/bin/sh
export PYTHONPATH=../everything:$PYTHONPATH
arg1=$1
arg2=$2
arg3=$3

python3 ../pythonscripts/mmd_gap_mortality_patient_year.py $arg1 $arg2 $arg3