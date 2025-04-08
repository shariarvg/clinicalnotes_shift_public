import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
import mmd_tools

ep = MIMICEndpoint()

diagnoses = pd.read_csv("../../../physionet.org/files/mimiciv/3.0/hosp/diagnoses_icd.csv.gz")

icd_version = 9

diagnoses = diagnoses[(diagnoses['icd_version']==icd_version)]

top_icd10_codes = diagnoses['icd_code'].value_counts()
codes = list(top_icd10_codes.index[:200])

year1 = 2009
year2 = 2012


model_org = "UFNLP/"
model_name = "gatortron-base"
model = model_org + model_name
summary = "mean"
max_length = 300
N = 100

data = []
for code in codes:
    
    diag_notes = ep.get_notes_diagnosis(code, icd_version)
    
    group1 = ep.get_notes_start_year(year1, notes = diag_notes)
    group2 = ep.get_notes_start_year(year2, notes = diag_notes)
    shape1 = group1.shape[0]
    shape2 = group2.shape[0]
    if shape1 < N or shape2 < N:
        print(group1)
        print(group2)
        print("---")
        continue
        
    group1 = group1.sample(N)
    group2 = group2.sample(N)
    
    emb1 = mmd_tools.get_doc_embeddings(list(group1['text']), model_name = model, summary = summary, max_length = max_length)
    emb2 = mmd_tools.get_doc_embeddings(list(group2['text']), model_name = model, summary = summary, max_length = max_length)
    
    _, _, _, mmd = mmd_tools.mmd_calc(emb1, emb2)
    
    #data.append([d, icd9, icd10, shape1, shape2, mmd])
    data.append([code, shape1, shape2, mmd])
    
#pd.DataFrame(data = data, columns = ["long_title", "code9", "code10", "shape9", "shape10", "MMD"]).to_csv(f"../../corresponding_diagnoses_icd_switch_{model_name}_{summary}_{str(max_length)}_N{N}.csv")
pd.DataFrame(data = data, columns = ["code", "shape9", "shape10", "MMD"]).to_csv(f"../../same_diagnosis_{year1}_{year2}_{model_name}_{summary}_{str(max_length)}_N{N}.csv")
    
    
