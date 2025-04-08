'''
Obtain the MMD between two datasets of notes from the same code (different time periods) as well as the null distribution
'''
import numpy as np
import pandas as pd
import sys 
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
import mmd_tools
import json

ep = MIMICEndpoint()

CODE = str(sys.argv[1])

notes = ep.get_notes_diagnosis(CODE, 10)


# nul testing
#notes = notes.sample(frac = 1)
#notes_1 = notes.iloc[:200]
#notes_2 = notes.iloc[200:]

all_mmd = []
all_mmds = []

for count in range(100):
# alternative testing
    notes_early = notes[(notes['start_year'] == 2015)]
    notes_late = notes[(notes['start_year'] == 2018)]

    notes_early = notes_early.sample(min(notes_early.shape[0], 200))
    notes_late = notes_late.sample(min(notes_late.shape[0], 200))

    #print("Shapes")
    print(notes_early.shape)
    print(notes_late.shape)

    X1 = mmd_tools.get_doc_embeddings(list(notes_early['text']), model_name = "UFNLP/gatortron-base")
    X2 = mmd_tools.get_doc_embeddings(list(notes_late['text']), model_name = "UFNLP/gatortron-base")


    mmd, mmds = mmd_tools.mmd_permutation_test(X1, X2, ret_null = True, number_bootstraps = 20)
    all_mmd.append(mmd)
    all_mmds.extend(mmds)

# Combine the data into a dictionary
data = {
    "all_mmd": list(all_mmd),
    "all_mmds": list(all_mmds)
}

# Save the data to a JSON file
with open(f"../../{CODE}_mmd_nulls_twodistributions.json", "w") as json_file:
    json.dump(data, json_file, indent=4)