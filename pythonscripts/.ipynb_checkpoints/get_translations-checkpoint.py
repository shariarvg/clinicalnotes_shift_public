import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
ep = MIMICEndpoint()
import numpy as np
import pandas as pd

V = "3"
V_prev = []#["1","0"]
codes = ['Z66', 'E874', 'J95851', 'Z515']
#codes = ["29620", "F329", "4019", "I10", "42833", "I5033", "V4986", "Z66"]
'''
codes = ['Z66',\
 'E874',\
 'J95851',\
 'Z515'\
 'J9602',\
 'R6521',\
 'K7200',\
 'R578',\
 'R570',\
 'J9600',\
 'G935',\
 'N170']
'''
versions = [10 for code in codes]
weights = np.ones(len(codes))/len(codes)
ms = MIMICSource(ep, "get_mixture", codes, versions, weights)

#note_ids_prev = set(pd.concat([pd.read_csv(f"../../note_ids_pre_translated_notes_V{V}.csv") for V in V_prev])['note_id'])

notes = ms.obtain_samples(2000)
#notes = notes[(notes['note_id'] not in note_ids_prev)].sample(5000)

print(notes.head())

notes[['note_id']].to_csv(f'../../note_ids_pre_translated_notes_V{V}.csv')

for count in range(0, notes.shape[0], 200):
    translated_notes = ep.get_translated_notes(list(notes.iloc[count:count+200]['text']))
    pd.DataFrame({"OG": list(notes.iloc[count:count+200]['text']), "New": translated_notes}).to_csv(f"../../translated_notes_{str(count)}_{str(count+200)}_V{V}.csv")
    
