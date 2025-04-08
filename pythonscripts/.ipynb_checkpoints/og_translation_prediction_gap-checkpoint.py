'''
Training on model on original notes, testing on Gemini translations
'''
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

## define parameters
V = "12"
max_length = 300
summary = 'mean'
pred_model = RandomForestClassifier
NOTES_TRAIN = 1000
NOTES_TEST = 1000

commit_hash = sys.argv[1]

'''
Obtain og and translated notes
'''
#note_ids = pd.concat([pd.read_csv("../../note_ids_pre_translated_notes_V0.csv"), pd.read_csv("../../note_ids_pre_translated_notes_V1.csv")])
note_ids = pd.read_csv("../../note_ids_pre_translated_notes_V3.csv")

ep = MIMICEndpoint()



og_notes = pd.merge(note_ids, ep.notes[['note_id', 'admission_in_30_days','death']], left_on = "note_id", right_on = "note_id") ## replace with a get_notes_note_ids()?

translated_notes = pd.DataFrame()
#translated_notes = pd.read_csv(f"../../translated_notes_{i}_{i+200}_V{3}.csv")
for i in range(0,2000,200):
    translated_notes = pd.concat([translated_notes, pd.read_csv(f"../../translated_notes_{i}_{i+200}_V3.csv")])
    
translated_notes[['admission_in_30_days','death']] = og_notes[['admission_in_30_days','death']].values
    
#for i in range(0,1000,200):
#    translated_notes = pd.concat([translated_notes, pd.read_csv(f"../../translated_notes_{i}_{i+200}_V1.csv")])

for task in ['admission_in_30_days', 'death']:
    for (model_org, model_name) in [("UFNLP/","gatortron-base"), ("","fine_tuned_gatortron"), ("", "sentence"),("", "concat"), ("","fine_tuned_gatortron_V3")]:
        model = model_org + model_name
        if model == "concat":
            model = ["UFNLP/gatortron-base", "fine_tuned_gatortron"]

        aucs_og = []
        aucs_trans = []
        aucs_train = []

        for count in range(50):
            translated_notes = translated_notes.sample(frac = 1)
            
            notes_train = list(translated_notes['OG'].iloc[:NOTES_TRAIN])
            notes_test_og = list(translated_notes['OG'].iloc[NOTES_TRAIN:])
            notes_test_trans = list(translated_notes['New'].iloc[NOTES_TRAIN:])
            
            '''
            get embeddings and outcomes
            '''
            emb_train = mmd_tools.get_doc_embeddings(notes_train, summary = 'mean', max_length = 300, model_name = model)
            emb_test_og = mmd_tools.get_doc_embeddings(notes_test_og, summary = 'mean', max_length = 300, model_name = model)
            emb_test_trans = mmd_tools.get_doc_embeddings(notes_test_trans, summary = 'mean', max_length = 300, model_name = model)

            '''
            train, get train AUC
            '''
            m = pred_model(max_depth = 5)
            m.fit(emb_train, translated_notes[task].iloc[:NOTES_TRAIN])
            probs_train = m.predict_proba(emb_train)
            aucs_train.append(roc_auc_score(translated_notes[task].iloc[:NOTES_TRAIN], probs_train[:,1]))
            
            probs_og = m.predict_proba(emb_test_og)
            aucs_og.append(roc_auc_score(translated_notes[task].iloc[NOTES_TRAIN:], probs_og[:,1]))

            '''
            eval, get AUC
            '''
            probs_translations = m.predict_proba(emb_test_trans)
            aucs_trans.append(roc_auc_score(translated_notes[task].iloc[NOTES_TRAIN:], probs_translations[:,1]))

        np.save(f"../experimentresults/og_performance_{task}_{model_name}_{max_length}_{summary}_V{V}.npy", np.array(aucs_og))
        np.save(f"../experimentresults/translation_performance_{task}_{model_name}_{max_length}_{summary}_V{V}.npy", np.array(aucs_trans))
        np.save(f"../experimentresults/train_performance_{task}_{model_name}_{max_length}_{summary}_V{V}.npy", np.array(aucs_train))
        
    
commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash

with open("../experimentresults/og_translation_prediction_gap.txt", 'w') as f:
    f.write("From og_translation_prediction_gap.py \n")
    f.write(commit_link)
    
print(f"Finished Running {os.path.basename(__file__)} V{V}")