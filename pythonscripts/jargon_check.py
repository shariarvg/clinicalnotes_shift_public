import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
import mmd_tools

ep = MIMICEndpoint()

V = 2
je = pd.read_csv(f"../../MIMIC_jargon_expansions_{V}.csv")

je = je[(je['Expansion'].apply(lambda x: len(x.split())) == 1)]

year1 = 2008
year2 = 2017

notes1 = ep.get_notes_start_year(year1, total_size = None)
notes2 = ep.get_notes_start_year(year2, total_size = None)

def first_N(note, N = 100):
    split = note.split()
    if len(split) <= 100:
        return note
    return " ".join(split[:100])

notes1['text'] = notes1['text'].apply(lambda x: first_N(x))
notes2['text'] = notes2['text'].apply(lambda x: first_N(x))

data = []

for (jarg, exp) in zip(je['Abbreviation'], je['Expansion']):
    count_j_1 = ep.count_notes_with_keyword(jarg, notes = notes1)
    count_j_2 = ep.count_notes_with_keyword(jarg, notes = notes2)
    count_e_1 = ep.count_notes_with_keyword(exp, notes = notes1)
    count_e_2 = ep.count_notes_with_keyword(exp, notes = notes2)
    
    data.append([jarg, exp, count_j_1, count_j_2, count_e_1, count_e_2])
    
save_name = f"../../jargon_expansion_counts_{year1}_{year2}_{V}.csv"
   
pd.DataFrame(data = data, columns = ["Jargon", "Expansion", "Count_jargon_"+str(year1), "Count_jargon_"+ str(year2), "Count_expansion_"+str(year1), "Count_expansion_"+str(year2)]).to_csv(save_name)
                 
             