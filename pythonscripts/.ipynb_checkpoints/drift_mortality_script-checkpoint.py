text = """
from drift_mortality_script.py
"""

import sys, os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath("../pythontools"))
sys.path.append(os.path.abspath("../experimenttools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
import auc
import random
import torch.nn as nn
import mmd_tools

ep = MIMICEndpoint()

commit_hash = sys.argv[1]

syears = np.arange(2008, 2021, 3)

model_name = "UFNLP/gatortron-base"
summary = "mean"
max_length = 100

save_name = "../experimentresults/gatortron_base_embeddings_"

for syear in syears:
    notes = ep.get_notes_start_year(syear, total_size = 100)
    embeddings = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = model_name, summary = summary, max_length = max_length)
    np.save(save_name+str(syear)+".npy", embeddings)
    

commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash
    
with open(save_name + "out.txt", 'w') as f:
    f.write(text + "\n")
    f.write(commit_link)
    
