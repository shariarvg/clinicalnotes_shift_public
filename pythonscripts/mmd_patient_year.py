import pandas as pd
import numpy as np

import torch
import scipy
import sys
from mmd_tools import get_doc_embeddings, mmd_permutation_test, power
# load model and tokenizer

from mimic_tools import MIMICEndpoint

patients = pd.read_csv("../physionet.org/files/mimiciv/3.0/hosp/patients.csv.gz")
admissions = pd.read_csv("../physionet.org/files/mimiciv/3.0/hosp/admissions.csv.gz")
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['length_of_stay'] =  (admissions['dischtime'] - admissions['admittime']).dt.total_seconds()/3600

ep = MIMICEndpoint()

CODE1 = str(sys.argv[1])
VERS1 = int(sys.argv[2])
CODE2 = str(sys.argv[3])
VERS2 = int(sys.argv[4])


notes = ep.get_notes_diagnosis(CODE1, VERS1)
notes = pd.concat([notes, ep.get_notes_diagnosis(CODE2, VERS2)])
notes = notes.merge(patients[['subject_id', 'anchor_age', 'anchor_year', 'anchor_year_group']], how = 'left')
notes = notes.merge(admissions[['hadm_id', 'admittime', 'length_of_stay']], how = 'left')
notes_first_year = notes[(notes["anchor_year"] == notes["admittime"].dt.year)]
notes_first_year_early = notes_first_year[(notes_first_year['anchor_year_group'] == '2008 - 2010')]
notes_first_year_late = notes_first_year[(notes_first_year['anchor_year_group'] == '2014 - 2016')]

print(power(notes_first_year_early['text'], notes_first_year_late['text']))