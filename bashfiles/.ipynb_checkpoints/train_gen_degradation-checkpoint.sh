#!/bin/sh
export PYTHONPATH=../everything:$PYTHONPATH

arg1=$1
arg2=$2
arg3=$3
arg4=$4
arg5=$5

python3 ../pythonscripts/get_gemini_diagnosis.py $arg1 $arg2 $arg3 $arg4
python3 ../pythonscripts/train_gen.py $arg1 $arg2 $arg3 $arg4
python3 ../pythonscripts/saved_note_mmd_logit_storer.py $arg1 $arg2 $arg3 $arg4
#python3 ../pythonscripts/mmd_and_degradation_by_embedding.py $arg1 $arg2 $arg3 $arg4 $arg5