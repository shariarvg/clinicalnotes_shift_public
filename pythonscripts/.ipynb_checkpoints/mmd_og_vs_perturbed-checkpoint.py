'''
MMD between a dataset and its perturbed version
'''
import numpy as np
import pandas as pd
import sys 
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
import mmd_tools
import json

V = "0p3"
ep = MIMICEndpoint()
SIZE = 30
codes = ['Z515', 'Z66', 'I10', 'F329']
mmds = []
mmdzs = []

for code in codes:
    
    notes2 = list(ep.get_notes_diagnosis(code, 10, total_size = SIZE)['text'])
    notes_perturbed = ep.get_translated_notes(notes2)
    del notes2
    notes = list(ep.get_notes_diagnosis(code, 10, total_size = SIZE)['text'])
    
    mmd, mmdz = mmd_tools.mmd_pipeline(notes, notes_perturbed, mmd_tools.mmd_permutation_test, ret = True, ret_sd = True)
    print(code)
    print(mmd)
    print(mmdz)
    print('--')
    mmds.append(mmd)
    mmdzs.append(mmdz)
    
pd.DataFrame({"Code": codes, "MMD": mmds, "MMDZ":mmdzs}).to_csv(f"../../mmd_og_vs_perturbed_v{V}.csv")
    
    
    
