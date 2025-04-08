import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource, MIMICMixtureSource
import pandas as pd
import numpy as np
import mmd_tools
import torch.nn as nn
import torch

ep = MIMICEndpoint()

#N_notes = 500
#func = "get_notes_start_year"
PARAM1 = 2016
PARAM2 = 2019
MAX_LENGTH = 100
DISEASE = 'htn'
save_name = "../experimentresults/mmd_and_degradation_"+DISEASE
commit_hash = sys.argv[1]

MODEL_EPOCH = 0
V_MODEL = 0
model_filepath = f"../../gatortron_{DISEASE}_classifier_chkpt_epoch{MODEL_EPOCH}_V{V_MODEL}.pt"

#ms1 = MIMICSource(ep, func, param1)
#ms2 = MIMICSource(ep, func, param2)

'''
%%%% CUSTOMIZE HOW NOTES ARE SOURCED
'''
#notes1 = ms1.obtain_samples(N_notes)
#notes2 = ms2.obtain_samples(N_notes)
#notes_eval = ms2.obtain_samples(N_notes)



note_ids = np.load(f"../experimentresults/note_id_{DISEASE}_Vunseen.pt.npy", allow_pickle = True)
notes = pd.DataFrame({"note_id": note_ids})
notes = pd.merge(notes, ep.notes[['note_id', 'text', 'hadm_id','start_year']], on = "note_id", how = "left")
answers = pd.read_csv(f"../experimentresults/{DISEASE}_diagnosis_answers_Vunseen.csv")
answers['task'] = answers['0'].str.contains("Yes").astype(int)
notes['task'] = answers['task'].values

notes1 = notes[(notes['start_year']==PARAM1)]
notes2 = notes[(notes['start_year']==PARAM2)]

emb1, logits1 = mmd_tools.get_doc_embeddings_and_prediction(list(notes1['text']), model_filepath = model_filepath, max_length = MAX_LENGTH)
emb2, logits2 = mmd_tools.get_doc_embeddings_and_prediction(list(notes2['text']), model_filepath = model_filepath, max_length = MAX_LENGTH)

mmd, mmdz = mmd_tools.mmd_permutation_test(emb1, emb2, ret = True, ret_sd = True)

loss = nn.CrossEntropyLoss()
loss1 = loss(logits1, torch.tensor(notes1['task'].values, dtype = torch.long))
loss2 = loss(logits1, torch.tensor(notes2['task'].values, dtype = torch.long))

commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash

with open(save_name + ".txt", 'w') as f:
    f.write(str((loss1-loss2).numpy()))
    f.write("\n")
    f.write(str(mmd))
    f.write("\n")
    f.write(str(mmdz))
    f.write('\n')
    f.write(commit_link)
    

