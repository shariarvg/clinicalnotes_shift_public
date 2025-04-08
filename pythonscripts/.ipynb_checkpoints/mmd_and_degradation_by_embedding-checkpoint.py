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
from sklearn.metrics import log_loss

ep = MIMICEndpoint()

V = "2"
V_notes = "1"

#N_notes = 500
#func = "get_notes_start_year"
commit_hash = sys.argv[1]
disease = sys.argv[2]
year1 = int(sys.argv[3])
year2 = int(sys.argv[4])
model_name = sys.argv[5]

device = 'cuda'

if model_name == "":
    model_at = f"../../experimentresults/{disease}_{year1}_{year2}_model.pt"
elif model_name == "bow":
    model_at =f"../../experimentresults/{disease}_{year1}_{year2}_bow"
else:
    model_at = f"../../experimentresults/{disease}_{year1}_{year2}_{model_name}_model.pt"
if disease in ['admission_in_30_days','death']:
    note_ids_at = f"../../experimentresults/cardio_{year1}_{year2}_note_id.npy"
else:
    note_ids_at = f"../../experimentresults/{disease}_{year1}_{year2}_note_id.npy"
note_ids = np.load(note_ids_at, allow_pickle = True)
notes = pd.DataFrame({"note_id": note_ids})
notes= pd.merge(notes, ep.notes, on = "note_id", how = "left")
save_name = f"../../experimentresults/{disease}_{year1}_{year2}_mmd_and_degradation_by_embedding_{model_name}"

if disease not in ['death', 'admission_in_30_days']:

    answers_at = f"../../experimentresults/{disease}_{year1}_{year2}_diagnosis_answers.csv"
    
    answers = pd.read_csv(answers_at)
    answers['task'] = answers['0'].str.contains("Yes").astype(int)
    notes['task'] = answers['task'].values
    
else:
    notes['task'] = notes[disease].values

notes_first5k_syear1 = notes[(notes['start_year']==year1)].iloc[:5000] ## this is actually a superset of the training dataset, because the real training dataset is imbalanced. If i take the MMD between the true training dataset and the unbalanced eval dataset, it will be high because of that class imbalance.
notes_train_pos = notes_first5k_syear1[(notes_first5k_syear1['task']==1)].iloc[:750]
notes_train_neg = notes_first5k_syear1[(notes_first5k_syear1['task']==0)].iloc[:750]
notes_train = pd.concat([notes_train_pos, notes_train_neg])

notes1_full = notes[(notes['start_year']==year1)].iloc[5000:]
notes2_full = notes[(notes['start_year']==year2)]

emb_first5k_syear1, _ = mmd_tools.get_doc_embeddings_and_prediction(list(notes_first5k_syear1['text']), model_filepath = model_at, model_base_name = model_name)
emb_train, _ = mmd_tools.get_doc_embeddings_and_prediction(list(notes_train['text']), model_filepath = model_at, model_base_name = model_name)


N_trials = 50
N_notes = 50

data = np.zeros((N_trials, 24))
loss_fn = nn.CrossEntropyLoss()
for i in range(N_trials):
    notes1 = notes1_full.sample(50)
    notes2 = notes2_full.sample(50)
    
    emb1, logits1 = mmd_tools.get_doc_embeddings_and_prediction(list(notes1['text']), model_filepath = model_at, model_base_name = model_name)
    emb2, logits2 = mmd_tools.get_doc_embeddings_and_prediction(list(notes2['text']), model_filepath = model_at, model_base_name = model_name)
    
    ## divergence between pre-balanced training dataset and 2014 eval notes
    mmd_f1, mmdq_f1, mmdz_f1 = mmd_tools.mmd_permutation_test(emb_first5k_syear1, emb1, ret= True, ret_sd = True, ret_quantile = True)
    mv_f1 = mauve.compute_mauve(p_features = emb_first5k_syear1, q_features = emb1).mauve
    
    ## divergence between pre-balanced training dataset and 2017 eval notes
    mmd_f2, mmdq_f2, mmdz_f2 = mmd_tools.mmd_permutation_test(emb_first5k_syear1, emb2, ret= True, ret_sd = True, ret_quantile = True)
    mv_f2 = mauve.compute_mauve(p_features = emb_first5k_syear1, q_features = emb2).mauve
    
    ## divergence between balanced training dataset and 2014 eval notes
    mmd_tr1, mmdq_tr1, mmdz_tr1 = mmd_tools.mmd_permutation_test(emb_train, emb1, ret= True, ret_sd = True, ret_quantile = True)
    mv_tr1 = mauve.compute_mauve(p_features = emb_train, q_features = emb1).mauve
    
    ## divergence between balanced training dataset and 2017 eval notes
    mmd_tr2, mmdq_tr2, mmdz_tr2 = mmd_tools.mmd_permutation_test(emb_train, emb2, ret= True, ret_sd = True, ret_quantile = True)
    mv_tr2 = mauve.compute_mauve(p_features = emb_train, q_features = emb2).mauve
    
    ## divergence between 2014 eval notes and 2017 eval notes
    mmd_12, mmdq_12, mmdz_12 = mmd_tools.mmd_permutation_test(emb1, emb2, ret= True, ret_sd = True, ret_quantile = True)
    mv_12 = mauve.compute_mauve(p_features = emb1, q_features = emb2).mauve
    
    if not isinstance(logits1, torch.Tensor): ## in this case, those logits are actually rfc probabilities
        loss1 = log_loss(notes1['task'], logits1)
        loss2 = log_loss(notes2['task'], logits2)
    else: 

        ## labels
        labels1 = torch.tensor(notes1["task"].values, dtype=torch.long, device=device)
        labels2 = torch.tensor(notes2["task"].values, dtype=torch.long, device=device)

        # loss
        loss1 = loss_fn(logits1, labels1).item()
        loss2 = loss_fn(logits2, labels2).item()
        
        auc1 = roc_auc_score(torch.sigmoid(logits1).numpy(), notes1['task'].to_numpy())
        auc2 = roc_auc_score(torch.sigmoid(logits2.numpy()), notes2['task'].to_numpy())
    
    ## data
    data[i] = [mmd_f1, mmdq_f1, mmdz_f1, mmd_tr1, mv_f1, mmd_f2, mmdq_f2, mmdz_f2, mv_f2, mmdq_tr1, mmdz_tr1, mv_tr1, mmd_tr2, mmdq_tr2, mmdz_tr2, mv_tr2, mmd_12, mmdq_12, mmdz_12, mv_12, loss1, loss2, auc1, auc2]
    
np.save(save_name + "_degradation_results.npy", data)


commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash

with open(save_name + ".txt", 'w') as f:
    f.write("mmd_and_degradation_by_embedding.py\n")
    f.write(commit_link)
    

