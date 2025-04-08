import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
ep = MIMICEndpoint()
import numpy as np
import mmd_tools
import torch
import pandas as pd

disease_dict = {'htn': 'hypertension', 'cardio': 'cardiovascular disease', 'dm2': 'diabetes', 'auto': 'autoimmune disease', 'death':'mortality'}

commit_hash = sys.argv[1]
disease = sys.argv[2]
year1 = int(sys.argv[3])
year2 = int(sys.argv[4])
disease_full = disease_dict[disease]



with open("../../api_key_gemini.txt", 'r') as f:
    api_key = f.readline()
    
import google.generativeai as genai

genai.configure(api_key=api_key)

# Select the Gemini model you want to use
model = genai.GenerativeModel("gemini-1.5-flash")

def get_diagnosis(note, model = model): 
    start = f"Based on this brief hospital course, does the patient have diagnosed {disease_full}? Answer yes or no. "
    return model.generate_content(start + note).text

def get_summaries(notes, model = model):
    return [get_summary(note, model) for note in notes]

note_id_filepath = f"../../experimentresults/{disease}_{year1}_{year2}_note_id.npy"
answer_filepath = f"../../experimentresults/{disease}_{year1}_{year2}_diagnosis_answers.csv"
save_name = f"../../experimentresults/{disease}_{year1}_{year2}"

## remove all notes previously seen, particularly those used for fine-tuning
'''
note_ids_already_used = np.load(note_id_filepath, allow_pickle = True)
notes = pd.DataFrame({"note_id": note_ids_already_used})
notes = pd.merge(notes, ep.notes, how = 'left', on = 'note_id')
ep.delete_notes(notes)

diagnosis_already_seen = list(pd.read_csv(answer_filepath)['0'])
'''

#notes_df = pd.concat([ep.get_notes_start_year(2016, 500), ep.get_notes_start_year(2019, 500)])
ms1 = MIMICSource(ep, "get_notes_start_year", year1)
ms2 = MIMICSource(ep, "get_notes_start_year", year2)

notes_train = ms1.obtain_samples(6000)
notes_unseen = ms2.obtain_samples(1000)
notes_df = pd.concat([notes_train, notes_unseen])

note_ids = np.array(notes_df['note_id'])
notes = list(notes_df['text'])
diagnosis = [get_diagnosis(note, model) for note in notes]

pd.Series(diagnosis).to_csv(answer_filepath)
np.save(note_id_filepath, note_ids)

commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash

with open(save_name + '.txt', 'w') as f:
    f.write(f"{disease_full}, {year1}-{year2}")
    f.write(commit_link)
