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

V = "1"

#N_notes = 500
#func = "get_notes_start_year"
commit_hash = sys.argv[1]
disease = sys.argv[2]
year1 = int(sys.argv[3])
year2 = int(sys.argv[4])

device = 'cuda'

model_at = f"../../experimentresults/{disease}_{year1}_{year2}_jargonlesstrain_V{V}_model.pt"
note_ids_at = f"../../experimentresults/{disease}_{year1}_{year2}_V{V}_note_id.npy"
note_ids = np.load(note_ids_at, allow_pickle = True)
notes = pd.DataFrame({"note_id": note_ids})
notes= pd.merge(notes, ep.notes, on = "note_id", how = "left")
save_name = f"../../experimentresults/{disease}_{year1}_{year2}_mmd_and_degradation_perturbed_by_embedding_V{V}"

if disease not in ['death', 'admission_in_30_days']:

    answers_at = f"../../experimentresults/{disease}_{year1}_{year2}_V{V}_diagnosis_answers.csv"
    
    answers = pd.read_csv(answers_at)
    answers['task'] = answers['0'].str.contains("Yes").astype(int)
    notes['task'] = answers['task'].values
    
else:
    notes['task'] = notes[disease].values

notes1_full = notes[(notes['start_year']==year1)].iloc[2000:]
notes2_full = notes[(notes['start_year']==year2)]


N_trials = 50
N_notes = 50

data = np.zeros((N_trials, 6))
loss_fn = nn.CrossEntropyLoss()
for i in range(20):
    notes1 = notes1_full.sample(50)
    notes2 = notes2_full.sample(50)
    
    emb1, logits1 = mmd_tools.get_doc_embeddings_and_prediction(list(notes1['text']), model_filepath = model_at)
    emb2, logits2 = mmd_tools.get_doc_embeddings_and_prediction(list(notes2['text']), model_filepath = model_at)
    
    mmd, mmdq, mmdz = mmd_tools.mmd_permutation_test(emb1, emb2, ret= True, ret_sd = True, ret_quantile = True)
    mv = mauve.compute_mauve(p_features = emb1, q_features = emb2).mauve

    labels1 = torch.tensor(notes1["task"].values, dtype=torch.long, device=device)
    labels2 = torch.tensor(notes2["task"].values, dtype=torch.long, device=device)

    # Compute loss
    loss1 = loss_fn(logits1, labels1).item()
    loss2 = loss_fn(logits2, labels2).item()
    
    data[i] = [mmd, mmdq, mmdz, mv, loss1, loss2]
    
np.save(save_name + "_degradation_results.npy", data)


commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash

with open(save_name + ".txt", 'w') as f:
    f.write("mmd_and_degradation_by_embedding.py\n")
    f.write(commit_link)
    

