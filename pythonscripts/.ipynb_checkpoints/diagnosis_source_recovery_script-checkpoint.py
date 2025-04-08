'''
Script for running source recovery experiments
'''

text = '''
Sources are pure diagnoses

For each row in output, there are columns code1...code10 (which represent the code of each source), eval_assn (which represents the code of the eval dataset), mmd1...mmd10 (which represent the mmd's of each ref against the eval), mauve1...mauve10 (which represent the mauve's of each ref against the eval)
'''

import sys, os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath("../pythontools"))
sys.path.append(os.path.abspath("../experimenttools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
import source_recovery
from source_recovery import get_mmds, get_mauves
import random

commit_hash = sys.argv[1]

with open("../../codes_to_use.txt", 'r') as f:
    codes = [line.rstrip() for line in f]

N_trials = 1000
N_sources = 10
N_notes = 100

#embedding information
model_name = "fine_tuned_gatortron_V2"
summary = "mean"
max_length = 300

save_name = "../experimentresults/diagnosis_source_recovery"

data = []
columns = [f"SourceCode{i}" for i in range(N_sources)] + ["EvalAssn"] + [f"MMD{i}" for i in range(N_sources)] + [f"Mauve{i}" for i in range(N_sources)] + ["MMDCorrect", "MauveCorrect"]

mmd_correct_counter = 0.0
mauve_correct_counter = 0.0
for count in range(N_trials):
    source_codes = random.sample(codes, N_sources)
    
    ep = MIMICEndpoint()
    
    sources = [MIMICSource(ep, "get_notes_diagnosis", code, 10) for code in source_codes]
    assignment, metrics = source_recovery.test_full_gen(sources, N_notes, model_name = model_name, summary = summary, max_length = max_length, method = [get_mmds, get_mauves])
    mmd_correct = int(np.argmin(metrics[0]) == assignment)
    mauve_correct = int(np.argmin(metrics[1]) == assignment)
    mmd_correct_counter += mmd_correct
    mauve_correct_counter += mauve_correct
    
    print(metrics)
    data.append(source_codes + [assignment] + metrics[0] + metrics[1] + [mmd_correct, mauve_correct])
    
pd.DataFrame(data = data, columns = columns).to_csv(save_name + ".csv")

mean_mmd_correct = mmd_correct_counter/N_trials
mean_mauve_correct = mauve_correct_counter/N_trials

commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash


with open(save_name + ".txt", 'w') as out_file:
    out_file.write(text)
    out_file.write("\n")
    out_file.write(f"MMD correct proportion: {mean_mmd_correct}\n")
    out_file.write(f"Mauve correct proportion: {mean_mauve_correct}\n")
    out_file.write(commit_link)
                                                        
