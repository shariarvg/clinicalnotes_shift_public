'''
Bunch of codes, want to calculate MMDz over a couple time periods
'''
import numpy as np
import pandas as pd
import sys 
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import mmd_tools
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import mauve

total_size = 50

codes = []
V = 0
mmdz_1517_1820 = []
mmdz_1418_1721 = []

ep = MIMICEndpoint()

for code in codes:
    notes_code = ep.get_notes_diagnosis(code, 10)
    
    ##15-17 vs. 18-20
    notes1 = ep.get_notes_start_year(2015, total_size = total_size, notes = notes_code)
    notes2 = ep.get_notes_start_year(2018, total_size = total_size, notes = notes_code)
    
    mmdz_1517_1820.append(mmd_tools.mmd_pipeline(notes1, notes2, mmd_permutation_test, ret_sd = True))
    
    del notes1
    del notes2
    
    ##14-18 vs. 17-21
    notes1 = ep.get_notes_start_year([2014, 2015, 2016], total_size = total_size, notes = notes_code)
    notes2 = ep.get_notes_start_year([2017, 2018, 2019], total_size = total_size, notes = notes_code)
    
    mmdz_1418_1721.append(mmd_tools.mmd_pipeline(notes1, notes2, mmd_permutation_test, ret_sd = True))
    
    del notes1
    del notes2
    
pd.DataFrame({"Code": codes, "mmdz1517_1820": mmdz_1517_1820, "mmdz1418_1721": mmdz_1418_1721}).to_csv(f"mmdz_by_code_and_gap_v{V}.csv")