'''
Generating a grid of AUC values for train test pairs.

https://www.notion.so/MMDz-Under-Dimension-Reduction-14402ba0ca078024a8fef9a22e2e88f3?pvs=4
'''


codes = [('29620', 'F329'), ('4019', 'I10'), ('42833', 'I5033'), ('V4986', 'Z66'), ('V667', 'Z515')]

dims = [2,5,10,25,100, float("inf")]

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
from mmd_tools import TextPCA
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import mauve

ep = MIMICEndpoint()
VERSION = "1p2"
out_array = np.zeros((len(codes), len(dims)))

for i, codepair in enumerate(codes):
    for j, dim in enumerate(dims):
        print(codepair[0])
        print(codepair[1])
        notes1 = ep.get_notes_diagnosis(codepair[0], 9, total_size = 100)
        notes2 = ep.get_notes_diagnosis(codepair[1], 10, total_size = 100)
        #notes1 = ep.get_notes_start_year(2015, total_size = 100, notes = notes)
        #notes2 = ep.get_notes_start_year(2018, total_size = 100, notes = notes)
        mmdz = mmd_tools.mmd_pipeline(notes1, notes2, mmd_tools.mmd_permutation_test, pca_embedding = TextPCA(dim), iterations = 20, ret_sd = True)
        out_array[i][j] = mmdz
        del notes1
        del notes2

np.save(f"../../mmd_dim_V{VERSION}.npy", out_array)
        