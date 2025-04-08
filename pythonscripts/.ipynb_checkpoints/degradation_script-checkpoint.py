'''
Script for running degradation experiments
'''

text = '''
Sources are just start years 2011 and 2019

Running from degradation_script.py
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
import grid
from featurization_tools import BOW, Transformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, mean_squared_error
import degradation

commit_hash = sys.argv[1]

ep = MIMICEndpoint()

sampler1 = MIMICSource(ep, "get_notes_start_year", 2011)
sampler2 = MIMICSource(ep, "get_notes_start_year", 2018)

N_samples = 5000

notes1_tr = sampler1.obtain_samples(N_samples)
notes2_tr = sampler2.obtain_samples(N_samples)
notes_te = sampler2.obtain_samples(N_samples)

for task in ['admission_in_30_days']:
    save_name = "../experimentresults/predict_degradation_"+task
    degradation.get_results(notes1_tr, notes2_tr, notes_te, BOW(), task, save_name)
