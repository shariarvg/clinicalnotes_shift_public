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
import mimic_tools

ep = MIMICEndpoint()

#N_notes = 500
#func = "get_notes_start_year"
commit_hash = sys.argv[1]
disease = sys.argv[2]
year1 = sys.argv[3]
year2 = sys.argv[4]
device = 'cuda'

for model_name in ['gpt2', 'gtron', 'bow']:

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
    save_name = f"../../experimentresults/{disease}_{year1}_{year2}_mmd_logit_storer_{model_name}"

    if disease not in ['death', 'admission_in_30_days']:

        answers_at = f"../../experimentresults/{disease}_{year1}_{year2}_diagnosis_answers.csv"

        answers = pd.read_csv(answers_at)
        answers['task'] = answers['0'].str.contains("Yes").astype(int)
        notes['task'] = answers['task'].values

    else:
        notes['task'] = notes[disease].values

    notes_train = notes[(notes['start_year']==year1)].iloc[:5000].reset_index()
    notes1_full = notes[(notes['start_year']==year1)].iloc[5000:].reset_index()
    notes2_full = notes[(notes['start_year']==year2)].reset_index()

    balanced_notes_train = ep.balanced(notes_train, 'task', 1500)

    notes1_full_lb = ep.largest_balanced(notes1_full, 'task')
    notes2_full_lb = ep.largest_balanced(notes2_full, 'task')

    emb_train, _ = mmd_tools.get_doc_embeddings_and_prediction(list(notes_train['text']), model_filepath = model_at, model_base_name = model_name)
    emb1, logits1 = mmd_tools.get_doc_embeddings_and_prediction(list(notes1_full['text']), model_filepath = model_at, model_base_name = model_name)
    emb2, logits2 = mmd_tools.get_doc_embeddings_and_prediction(list(notes2_full['text']), model_filepath = model_at, model_base_name = model_name)

    emb1_lb = emb1[notes1_full_lb.index, :]
    emb2_lb = emb2[notes1_full_lb.index, :]
    emb_train_lb = emb_train[balanced_notes_train.index, :]


    mmds_tr1_balanced = list(mmd_tools.mmd_permutation_test(emb_train_lb, emb1_lb, ret = True, ret_sd = True, ret_quantile = True))
    mmds_tr2_balanced = list(mmd_tools.mmd_permutation_test(emb_train_lb, emb2_lb, ret = True, ret_sd = True, ret_quantile = True))
    mmds_tr1= list(mmd_tools.mmd_permutation_test(emb_train, emb1, ret = True, ret_sd = True, ret_quantile = True))
    mmds_tr2 = list(mmd_tools.mmd_permutation_test(emb_train, emb2, ret = True, ret_sd = True, ret_quantile = True))
    mmds_12 = list(mmd_tools.mmd_permutation_test(emb1, emb2, ret = True, ret_sd = True, ret_quantile = True))
    #mmds_tr1lb = list(mmd_tools.mmd_permutation_test(emb_train, emb1, ret = True, ret_sd = True, ret_quantile = True))

    #if isinstance(logits1, np.ndarray):
    np.save(save_name + "_1.npy", logits1)
    np.save(save_name + "_2.npy", logits2)
    np.save(save_name + "_balanced_1.npy", logits1[notes1_full_lb.index, :])
    np.save(save_name + "_balanced_2.npy", logits2[notes2_full_lb.index, :])
    '''
    else:
        np.save(save_name + "_1.npy", logits1.cpu().numpy())
        np.save(save_name + "_2.npy", logits2.cpu().numpy())
        np.save(save_name + "_balanced_1.npy", logits1.cpu().numpy()[notes1_full_lb.index, :])
        np.save(save_name + "_balanced_2.npy", logits2.cpu().numpy()[notes2_full_lb.index, :])
    '''
    np.save(save_name+"_mmds.npy", np.array(mmds_tr1_balanced + mmds_tr2_balanced + mmds_tr1 + mmds_tr2 + mmds_12))


    commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash

    with open(save_name + ".txt", 'w') as f:
        f.write("mmd_and_degradation_by_embedding.py\n")
        f.write(commit_link)


