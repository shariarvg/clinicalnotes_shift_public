#!/bin/sh
export PYTHONPATH=../everything:$PYTHONPATH

arg1=$1

python3 ../pythonscripts/train_cardio.py $arg1