'''
Template script for running mixture experiments
'''

text = """
Mixtures of 2, implemented in mixture_script.py
"""

import sys, os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath("../pythontools"))
sys.path.append(os.path.abspath("../experimenttools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
from mixture import get_result

import random

commit_hash = sys.argv[1]

with open("../../codes_to_use.txt", 'r') as f:
    codes = [line.rstrip() for line in f]

N_trials = 500
N_weights_per_trial = 10
N_sources = 10
N_notes = 100

model_name = "fine_tuned_gatortron_V2"
summary = "mean"
max_length = 300

save_name = "../experimentresults/los_mixtures_of_2"

data = []
#columns = [f"Code{i}" for i in range(N_codes_per_source)] + [f"Weight_{a}_{b}" for b in range(N_codes_per_source) for a in range(N_sources)] + ["MMD", "MAUVE"]
columns = ["Trial","Weight1", "Weight2", "MMD", "MAUVE"]


mmd_correct_counter = 0.0
mauve_correct_counter = 0.0
for count in range(N_trials):
    for weight1 in np.arange(0,1,1.0/N_weights_per_trial):
        ep = MIMICEndpoint()
        source1 = MIMICSource(ep, "get_notes_los_greater_than", 4)
        source2 = MIMICSource(ep, "get_notes_los_less_than", 5)
        mmd, mauve = get_result([source1, source2], [weight1, 1- weight1], 0, N_notes, model_name, summary, max_length)
        data.append([count] + [weight1, 1-weight1] + [mmd, mauve])
    
pd.DataFrame(data = data, columns = columns).to_csv(save_name + ".csv")

commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash


with open(save_name + ".txt", 'w') as out_file:
    out_file.write(text)
    out_file.write("\n")
    out_file.write(commit_link)
                                     