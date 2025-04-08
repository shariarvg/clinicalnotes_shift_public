#! /bin/bash
export PYTHONPATH=../everything:$PYTHONPATH

# Get the commit hash from the command line argument or use the latest commit
if [ -z "$1" ]; then
    COMMIT_HASH=$(git rev-parse HEAD)
else
    COMMIT_HASH=$1
fi

echo "Running distributional comparison with commit hash: $COMMIT_HASH"

python3 ../pythonscripts/distributional_comparison.py --commit_hash $COMMIT_HASH