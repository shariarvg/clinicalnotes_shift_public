#!/bin/sh
export PYTHONPATH=../everything:$PYTHONPATH

arg1=$1

python3 ../pythonscripts/mse_script.py $arg1