#!/bin/sh
export PYTHONPATH=../everything:$PYTHONPATH

arg1=$1
arg2=$2
arg3=$3
arg4=$4

python3 ../pythonscripts/get_gemini_diagnosis.py $arg1 $arg2 $arg3 $arg4