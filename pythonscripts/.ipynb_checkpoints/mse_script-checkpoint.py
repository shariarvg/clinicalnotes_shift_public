'''
Script for running auc experiments
'''

text = '''
Sources are from length of stay

For each row in output, there are columns eval_assn (which represents the year of the eval dataset), mmd{i}..., mauve{i}... mse{i}...

Coming from mse_script.py
'''

import sys, os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath("../pythontools"))
sys.path.append(os.path.abspath("../experimenttools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
import auc
import random

commit_hash = sys.argv[1]


#with open("../../codes_to_use.txt", 'r') as f:
#    codes = [line.rstrip() for line in f]
   
N_trials = 500
N_notes = 100
N_sources = 10

#embedding information
model_name = "fine_tuned_gatortron_V2"
summary = "mean"
max_length = 300

save_name = "../experimentresults/losgroup_los"

lefts = [0, 3, 10, 20, 'inf']
data = []
columns = ["Assignment"] + [f"MMD{i}" for i in range(N_sources)] + [f"Mauve{i}" for i in range(N_sources)] + [f"MSE{i}" for i in range(N_sources)]

for count in range(N_trials):
    ep = MIMICEndpoint()
    
    sources = [MIMICSource(ep, "get_notes_los_range", lefts[i], lefts[i+1]) for i in range(len(lefts)-1)]
    i = random.randint(0,len(lefts)-1)
    eval_source = MIMICSource(ep, "get_notes_los_range", lefts[i], lefts[i+1])
    
    mmds, mauves, mses = auc.get_all_notes_metrics_mses(sources, eval_source, N_notes, model_name, summary, max_length, 'length_of_stay')
    
    data.append([i] + mmds + mauves + mses)
    

pd.DataFrame(data = data, columns = columns).to_csv(save_name + ".csv")
    
commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash

with open(save_name + ".txt", 'w') as out_file:
    out_file.write(text)
    out_file.write("\n")
    out_file.write(commit_link)
                                   