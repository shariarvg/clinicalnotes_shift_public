'''
No fine-tuned architectures
'''

import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource, MIMICMixtureSource
import pandas as pd
import numpy as np
import mmd_tools
import torch.nn as nn
import torch
from featurization_tools import BOW, Transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import mauve

ep = MIMICEndpoint()

V = "0"

N_notes = 500
func = "get_notes_start_year"
PARAM1 = 2008
PARAM2 = 2019
task = "admission_in_30_days"
save_name = "../experimentresults/mmd_and_degradation_by_embedding_"+task+f"_V{V}"
commit_hash = sys.argv[1]

ms1 = MIMICSource(ep, func, PARAM1)
ms2 = MIMICSource(ep, func, PARAM2)

notes_train = ms1.obtain_samples(1000)
notes1_full = ms1.obtain_samples(500)
notes2_full = ms2.obtain_samples(500)

featurizers = [BOW(0.1, 0.9), BOW(0.05, 0.95), BOW(0.5, 0.7)] + [Transformer(model_name = model, summary = s, max_length = m) for m in [100,300] for s in ['mean', 'first'] for model in ['gpt2', 'UFNLP/gatortron-base', 'fine_tuned_gatortron_V2']]

data = np.zeros((len(featurizers), 10))
for i in range(20):
    notes1 = notes1_full.sample(50)
    notes2 = notes2_full.sample(50)
    
    for i, featurizer in enumerate(featurizers):
        emb_train = featurizer.transform(list(notes_train['text'])) #fitting
        emb1 = featurizer.transform(list(notes1['text']))
        emb2 = featurizer.transform(list(notes2['text']))

        rfc = RandomForestClassifier()
        rfc.fit(emb_train, notes_train[task])

        probs1 = rfc.predict_proba(emb1)
        probs2 = rfc.predict_proba(emb2)
        auc1 = roc_auc_score(notes1[task], probs1[:,1])
        auc2 = roc_auc_score(notes2[task], probs2[:,1])
        
        mmd_1, mmdq_1, mmdz_1 = mmd_tools.mmd_permutation_test(emb_train, emb1, ret = True, ret_sd = True, ret_quantile = True)
        m_1 = mauve.compute_mauve(p_features = emb_train, q_features = emb1).mauve

        mmd_2, mmdq_2, mmdz_2 = mmd_tools.mmd_permutation_test(emb_train, emb2, ret = True, ret_sd = True, ret_quantile = True)
        m_2 = mauve.compute_mauve(p_features = emb_train, q_features = emb2).mauve

        data[i] = [auc1, auc2, mmd_1, mmdz_1, mmdq_1, m_1, mmd_2, mmdz_2, mmdq_2, m_2]

np.save(save_name + ".npy", data)
with open(save_name+"_featurizers.txt", 'w') as f:
    for featurizer in featurizers:
        f.write(featurizer.name + "\n")

commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash

with open(save_name + ".txt", 'w') as f:
    f.write("mmd_and_degradation_by_embedding_mimic_task.py\n")
    f.write(commit_link)
    

