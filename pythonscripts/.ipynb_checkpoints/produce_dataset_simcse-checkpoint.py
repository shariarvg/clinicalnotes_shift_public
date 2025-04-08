import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource

df_notes = pd.DataFrame()

V = "1"

for i in range(0, 1000, 200):
    df_notes = pd.concat([df_notes, pd.read_csv(f"../../translated_notes_{i}_{i+200}_V{V}.csv")])
    
df_notes = df_notes.rename(columns = {"OG": "Anchor", "New": "Positive"})



note_ids = pd.read_csv(f'../../note_ids_pre_translated_notes_V{V}.csv')

def switch_columns_with_probability(df):
    # Generate a random boolean mask with 50% True (independent for each row)
    switch_mask = np.random.rand(len(df)) < 0.5
    
    # Apply the mask to swap values
    df.loc[switch_mask, ['Anchor', 'Positive']] = df.loc[switch_mask, ['Positive', 'Anchor']].values
    return df

# Apply the function
df_notes = switch_columns_with_probability(df_notes.copy())


'''
Obtaining hard negatives
'''

ep = MIMICEndpoint()
note_ids = note_ids.merge(ep.notes[['note_id','hadm_id']], how = 'left', on = 'note_id')

codes = ["29620", "F329", "4019", "I10", "42833", "I5033", "V4986", "Z66"]

diagnoses = {}
for i in range(df_notes.shape[0]):
    hadm_id = note_ids.iloc[i]['hadm_id']
    d = set(ep.diagnoses[(ep.diagnoses['hadm_id'] == hadm_id)]['icd_code'].unique())
    diagnoses[i] = d
    
def comparison(i, j):
    return len(diagnoses[i].intersection(diagnoses[j]))

hard_negatives = []
for i in range(df_notes.shape[0]):
    for j in np.random.permutation(df_notes.shape[0]):
        if i == j:
            continue
        if comparison(i,j) > 0:
            hard_negatives.append(df_notes.iloc[j]['Anchor'])
            break
    if len(hard_negatives) != i+1:
        hard_negatives.append(df_notes.iloc[i-1]['Anchor'])

df_notes['HardNegative'] = hard_negatives
df_notes.to_csv(f"../../training_simcse_V{V}.csv")
    


                          
                          