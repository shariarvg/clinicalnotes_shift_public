#!/bin/sh
export PYTHONPATH=../everything:$PYTHONPATH

python3 ../pythonscripts/fine_tune_readmission.py $arg1