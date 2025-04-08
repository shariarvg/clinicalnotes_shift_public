import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("../pythontools"))
import mmd_tools
import torch
from torch import nn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

V = "0"
task = "death"

note_ids = pd.read_csv(f'note_ids_pre_translated_notes_V{V}.csv')

notes = pd.DataFrame()
for count in range(0, 1000, 200):
    notes_new = pd.read_csv(f"translated_notes_{str(count)}_{str(count+200)}_V{V}.csv")
    notes = pd.concat([notes, notes_new])
    
notes = pd.concat([note_ids, notes], axis = 1)
ep = MIMICEndpoint()
notes = pd.merge(notes, ep.notes[['note_id', task]], how = 'left')

cv = CountVectorizer(min_df = 0.1, max_df = 0.95)
new_embeddings= mmd_tools.get_doc_embeddings(list(notes['New']))
new_cv_features = cv.fit_transform(notes['New'])


N_TRAIN = 200
N_EVAL = 200
mmds = []
mmdz_s = []
scores = []
for count in range(N_TRIALS):
    ##Randomly pick 120 notes for training and testing
    index = np.arange(0,1000).sample(N_TRAIN+N_EVAL)
    it_emb = new_embeddings[index]
    it_feat = new_cv_features[index]
    
    it_task = notes[task].iloc[index,:]
    it_train_task = it_task.iloc[:N_TRAIN]
    it_test_task = it_task.iloc[N_TRAIN:]
    
    
    it_train_emb = it_emb[:N_TRAIN]
    it_test_emb = it_emb[N_TRAIN]
    it_train_feat = it_feat[:N_TRAIN]
    it_test_feat = it_feat[N_TEST:]
    
    mmd, mmd_z = mmd_tools.mmd_permutation_test(it_train_emb, it_test_emb, ret = True, ret_sd = True)
    mmds.append(mmd)
    mmdz_s.append(mmd_z)
    
    lm = LogisticRegression()
    lm.fit(it_train_feat, it_train_task)
    
    scores.append(accuracy_score(lm.predict(it_test_feat, it_test_task)))
    
