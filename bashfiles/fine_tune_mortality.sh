#!/bin/sh
export PYTHONPATH=../everything:$PYTHONPATH

python3 ../pythonscripts/fine_tune_mortality.py $arg1