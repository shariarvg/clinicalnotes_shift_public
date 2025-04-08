import pandas as pd
import numpy as np
import sys, os
from datetime import datetime
sys.path.append(os.path.abspath("../pythontools"))
sys.path.append(os.path.abspath("../experimenttools"))
import json

from mimic_tools import MIMICEndpoint

from source_pairs_degradation import gen_source_pair_deg_results

commit_hash = sys.argv[1]
commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash

N_trials = 100
N_train = 100
N_test = 100
task = "death_in_30_days"
V = 0
ke = "has_Z515"
val1 = 1
val2 = 0
sourcer = "get_notes_diagnosis"
param1 = "I5033"
param2 = "I10"

t_start = datetime.now()

out = "../../source_pairs_degradation_metric_groups.json"

gen_source_pair_deg_results(ke, val1, val2, sourcer, param1, param2, N_trials, N_train, N_test, task, V, "../../source_pairs_degradation_"+commit_hash)

if os.path.exists(out):
    with open(out, 'r') as f:
        data = json.load(f)
else:
    data = []

# Convert datetime objects to strings for JSON serialization
t_end = datetime.now()
runtime_seconds = (t_end - t_start).seconds

config = {
    "N_trials": N_trials,
    "N_train": N_train,
    "N_test": N_test,
    "task": task,
    "V": V,
    "ke": ke,
    "val1": val1,
    "val2": val2,
    "sourcer": sourcer,
    "param1": param1,
    "param2": param2,
    "hash": commit_hash,
    "link": commit_link,
    "time_finished": t_end.isoformat(),  # Convert to ISO format string
    "seconds_runtime": runtime_seconds
}

data.append(config)

with open(out, 'w') as f:
    json.dump(data, f, indent=4)