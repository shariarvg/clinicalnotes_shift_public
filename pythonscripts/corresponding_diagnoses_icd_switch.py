import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
import mmd_tools

ep = MIMICEndpoint()
'''
di = pd.read_csv("../../../physionet.org/files/mimiciv/3.0/hosp/d_icd_diagnoses.csv.gz")


# Example series
s = di['long_title']

# Get elements that appear more than once


duplicates = s[s.duplicated()].unique()
'''

gem = pd.read_csv("../../icd9toicd10cmgem.csv")
icd9s = list(gem['icd9cm'])
icd10s = list(gem['icd10cm'])


model_org = "UFNLP/"
model_name = "gatortron-base"
model = model_org + model_name
summary = "mean"
max_length = 300
N = 100

data = []
for (icd9, icd10) in zip(icd9s, icd10s):
    '''
    try:
        icd9 = di[((di['long_title']==d)&(di['icd_version'] == 9))]['icd_code'].iloc[0]
        icd10 = di[((di['long_title']==d)&(di['icd_version'] == 10))]['icd_code'].iloc[0]
    except:
        continue
    '''
    
    group1 = ep.get_notes_diagnosis(icd9, 9)
    group2 = ep.get_notes_diagnosis(icd10, 10)
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
    
    mmd = mmd_tools.mmd_permutation_test(emb1, emb2, ret_sd = True)
    
    #data.append([d, icd9, icd10, shape1, shape2, mmd])
    data.append([icd9, icd10, shape1, shape2, mmd])
    
#pd.DataFrame(data = data, columns = ["long_title", "code9", "code10", "shape9", "shape10", "MMD"]).to_csv(f"../../corresponding_diagnoses_icd_switch_{model_name}_{summary}_{str(max_length)}_N{N}.csv")
pd.DataFrame(data = data, columns = ["code9", "code10", "shape9", "shape10", "MMD"]).to_csv(f"../../corresponding_diagnoses_icd_switch_{model_name}_{summary}_{str(max_length)}_N{N}_gem_mmdz.csv")
    
    
