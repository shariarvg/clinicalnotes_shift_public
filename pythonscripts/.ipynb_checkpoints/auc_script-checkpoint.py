'''
Script for running auc experiments
'''

text = '''
Sources are codes

For each row in output, there are columns eval_assn (which represents the year of the eval dataset), mmd{i}..., mauve{i}... auc{i}...

Coming from auc_script.py
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


with open("../../codes_to_use.txt", 'r') as f:
    codes = [line.rstrip() for line in f]
   
N_trials = 500
N_notes = 100
N_sources = 10

#embedding information
model_name = "fine_tuned_gatortron_V2"
summary = "mean"
max_length = 300

save_name = "../experimentresults/diagnosisgroup_recovery_readmission"

data = []
columns = [f"MMD{i}" for i in range(N_sources)] + [f"Mauve{i}" for i in range(N_sources)] + [f"AUC{i}" for i in range(N_sources)]

for count in range(N_trials):
    ep = MIMICEndpoint()
    
    source_codes = random.sample(codes, N_sources)
    eval_code = random.sample(codes, 1)[0]
    
    sources = [MIMICSource(ep, "get_notes_diagnosis", code, 10) for code in source_codes]
    eval_source = MIMICSource(ep, "get_notes_diagnosis", eval_code, 10)
    
    mmds, mauves, aucs = auc.get_all_notes_metrics_aucs(sources, eval_source, N_notes, model_name, summary, max_length, 'admission_in_30_days')
    
    data.append(mmds + mauves + aucs)
    

pd.DataFrame(data = data, columns = columns).to_csv(save_name + ".csv")
    
commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash

with open(save_name + ".txt", 'w') as out_file:
    out_file.write(text)
    out_file.write("\n")
    out_file.write(commit_link)
                                   