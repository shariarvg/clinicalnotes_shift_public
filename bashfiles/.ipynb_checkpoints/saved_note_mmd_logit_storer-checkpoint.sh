#!/bin/sh
export PYTHONPATH=../everything:$PYTHONPATH


arg1=$1

python3 ../pythonscripts/saved_note_mmd_logit_storer.py $arg1