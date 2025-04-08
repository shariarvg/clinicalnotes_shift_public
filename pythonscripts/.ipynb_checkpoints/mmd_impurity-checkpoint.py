'''
Checking to see if the MMD of a [set of notes from code A] and [a mixture of notes from code A and code B] is correctly aligned with how much weight is given to code B in the mixture
'''

import nltk
import pandas as pd
import numpy
import sys 
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
import mmd_tools

ep = MIMICEndpoint()

V = "3b"

step = 0.01
weights = np.arange(0, 1+step, step)
mmds = []
mmd_quantiles = []
weights_used = []
N = 200

source1 = MIMICSource(ep, "get_notes_diagnosis", "Z515", 10)
source2 = MIMICSource(ep, "get_notes_version", 10)

model = "fine_tuned_gatortron_V2"
summary = "first"
max_length = 100

for w in weights:
    notesP = source1.obtain_samples(N)
    notesM = pd.concat([source1.obtain_samples(int(N*w)), source2.obtain_samples(int(N-N*w))])
    if notesP.shape[0] ==0 or notesM.shape[0] == 0:
        continue
    embP = mmd_tools.get_doc_embeddings(list(notesP['text']), model_name = model, summary = summary, max_length = max_length)
    embM = mmd_tools.get_doc_embeddings(list(notesM['text']), model_name = model, summary = summary, max_length = max_length)
    mmd, quantile = mmd_tools.mmd_permutation_test(embP, embM, ret = True, ret_quantile = True)
    mmds.append(mmd)
    mmd_quantiles.append(quantile)
    weights_used.append(w)
    
df = pd.DataFrame({"Weight": weights_used, "MMD": mmds, "MMDQuantile": mmd_quantiles})
df.to_csv(f"../../mmd_impurity_{N}_V{V}.csv")

print(f"File {os.path.basename(__file__)} V{V}")
    