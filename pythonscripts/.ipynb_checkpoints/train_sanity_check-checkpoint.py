import numpy as np
import pandas as pd
import sys 
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource, MIMICMultiSource
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
import mmd_tools
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, mean_squared_error
import mauve
from featurization_tools import BOW, Transformer

codes = ['Z66',\
 'E874',\
 'J95851',\
 'Z515',\
 'J9602',\
 'R6521',\
 'K7200',\
 'R578',\
 'R570',\
 'J9600',\
 'G935',\
 'N170']

pred_model = RandomForestClassifier

ep = MIMICEndpoint()

note_ids = pd.read_csv("../../note_ids_pre_translated_notes_V3.csv")
og_notes = pd.merge(note_ids, ep.notes[['note_id', 'death']], left_on = "note_id", right_on = "note_id", how = 'left') ## replace with a get_notes_note_ids()?

translated_notes = pd.DataFrame()
for i in range(0,2000,200):
    translated_notes = pd.concat([translated_notes, pd.read_csv(f"../../translated_notes_{i}_{i+200}_V3.csv")])
    
translated_notes['death'] = list(og_notes['death'].values)

print(og_notes[['death']].head())
print(translated_notes[['death']].head())


#ms = MIMICSource(ep, "get_mixture", codes, [10 for code in codes], [1.0/len(codes) for code in codes])
notes_train = list(translated_notes['OG'].iloc[:1000])#ms.obtain_samples(200)
m = pred_model(max_depth = 5)
emb_train = mmd_tools.get_doc_embeddings(notes_train, summary = 'mean', max_length = 300, model_name = "fine_tuned_gatortron_V2")

m.fit(emb_train, og_notes['death'].iloc[:1000])

notes_test = list(translated_notes['OG'].iloc[1000:])#ms.obtain_samples(200)
pred_prob = m.predict_proba(mmd_tools.get_doc_embeddings(notes_test, summary = 'mean', max_length = 300, model_name = "fine_tuned_gatortron_V2"))
print(roc_auc_score(translated_notes['death'].iloc[1000:], pred_prob[:,1]))

print((translated_notes['death'] - og_notes['death']).sum())