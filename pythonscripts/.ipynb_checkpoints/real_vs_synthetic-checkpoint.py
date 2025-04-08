import numpy as np
import pandas as pd
import sys 
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
import mmd_tools

V = "sentence"

ep = MIMICEndpoint()

from datasets import load_dataset

synthetic_ds = pd.DataFrame(load_dataset("starmpcc/Asclepius-Synthetic-Clinical-Notes")['train'])

embedding_model = "sentence"

n_notes = 5
n_iterations = 50

mmd_term1 = np.zeros((2020-2009, n_iterations))
mmd_term2 = np.zeros((2020-2009, n_iterations))
mmd_term3 = np.zeros((2020-2009, n_iterations))
mmds = np.zeros((2020-2009, n_iterations))

for sy in range(2009, 2020, 1):
    for count in range(n_iterations):
        real_notes = ep.get_notes_start_year(sy, n_notes)
        synthetic_notes = synthetic_ds['note'].sample(n_notes)
        
        mmd1, mmd2, mmd3 = mmd_tools.mmd_pipeline(list(real_notes), list(synthetic_notes), mmd_tools.mmd_calc, model_name = embedding_model)
        
        mmd_term1[sy-2009][count] = mmd1
        mmd_term2[sy-2009][count] = mmd2
        mmd_term3[sy-2009][count] = mmd3
        mmds[sy-2009][count] = mmd1 + mmd2 - 2 * mmd3
        
np.save(f"../../real_v_synthetic_mmd1_V{V}.npy", mmds)
for i, a in enumerate([mmd_term1, mmd_term2, mmd_term3]):
    np.save(f"../../real_v_synthetic_mmd{i+1}_V{V}.npy", a)
np.save(f"../../real_v_synthetic_mmd_V{V}.npy", mmds)

        
        
        
        
        
    