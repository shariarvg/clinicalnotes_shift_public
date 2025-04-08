'''
Tempalte script for running ordinal shift validation experiments
'''

text = """
Threshold Z-Statistic, implemented between diagnosis mixtures
4 codes per source, each source has the same codes but has different weights coming from dirichlet(1,1,1,1)
10 sources, 100 notes
"""


import sys, os
import random
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath("../pythontools"))
sys.path.append(os.path.abspath("../experimenttools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource, MIMICMixtureSource
from mixture import get_result
from osv import get_all_notes_metrics_divs
import time

start = time.time()

commit_hash = sys.argv[1]

with open("../../codes_to_use.txt", 'r') as f:
    codes = [line.rstrip() for line in f]
    
N_codes_per_source = 4
N_trials = 100
N_sources = 10
N_notes = 100

model_name = "fine_tuned_gatortron_V2"
summary = "mean"
max_length = 100

divergence_endpoint = MIMICEndpoint()
divergence_fn = divergence_endpoint.count_threshold

save_name = f"../experimentresults/osv_threshold_z_{N_sources}_mixtures_{N_codes_per_source}_codes_v2"

data = []
columns = ["Source_Ind"] + [f"MMD{s}" for s in range(N_sources)] + [f"MAUVE{s}" for s in range(N_sources)] + [f"Div{s}" for s in range(N_sources)]

for count in range(N_trials):
    codes_for_trial = random.sample(codes, N_codes_per_source)
    
    all_weights = np.random.dirichlet([1 for code in codes_for_trial], N_sources)
    
    ep = MIMICEndpoint()
    
    sources = [MIMICMixtureSource([MIMICSource(ep, "get_notes_diagnosis", code, 10) for code in codes_for_trial], weights) for weights in all_weights]
    
    source_ind, mmds, mauves, divs = get_all_notes_metrics_divs(sources, N_notes, model_name, summary, max_length, divergence_fn)
    
    data.append([source_ind] + mmds + mauves + divs)
    

commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash

pd.DataFrame(data = data, columns = columns).to_csv(save_name + ".csv")

end = time.time()

with open(save_name + ".txt", 'w') as out_file:
    out_file.write(text)
    out_file.write("\n")
    out_file.write(f"Time: {end-start}\n")
    out_file.write(commit_link)