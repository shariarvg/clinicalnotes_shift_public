'''
Script for running grid experiments
'''

text = '''
Sources are length of stay groups

Output is a grid

Coming from grid_script.py
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

lefts = [0, 3, 10, 20, 'inf']
N_trials_per_pair = 100
featurizer = Transformer()
task = "length_of_stay"
N_notes = 100

commit_hash = sys.argv[1]

grid_arr = np.zeros((len(lefts)-1, len(lefts)-1))

for i in range(len(lefts[:-1])):
    for j in range(len(lefts[:-1])):
        ep = MIMICEndpoint()
        grid_arr[i][j] = grid.iteration(MIMICSource(ep, "get_notes_los_range", lefts[i], lefts[i+1]), MIMICSource(ep, "get_notes_los_range", lefts[j], lefts[j+1]), N_notes, Transformer(), 'length_of_stay', False, mean_squared_error)
        
save_name = "../experimentresults/grid_los"

np.save(save_name + ".npy", grid_arr)

commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash

with open(save_name + ".txt", 'w') as out_file:
    out_file.write(text)
    out_file.write("\n")
    out_file.write(commit_link) 
