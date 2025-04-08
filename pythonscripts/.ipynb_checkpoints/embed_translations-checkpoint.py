import pandas as pd
import numpy as np
import torch
import sys, os
sys.path.append(os.path.abspath("../pythontools"))
import mmd_tools
V = "0"

model = "fine_tuned_gatortron_V2"

notes = pd.DataFrame()
V = "0"
for count in range(0, 1000, 200):
    notes_new = pd.read_csv(f"../../translated_notes_{str(count)}_{str(count+200)}_V{V}.csv")
    notes = pd.concat([notes, notes_new])
    
old_embeddings= mmd_tools.get_doc_embeddings(list(notes['OG']), model_name = model)
new_embeddings= mmd_tools.get_doc_embeddings(list(notes['New']), model_name = model)

np.save(f"../../old_embeddings_V{V}_finetuned.npy", old_embeddings)
np.save(f"../../new_embeddings_V{V}_finetuned.npy", new_embeddings)

