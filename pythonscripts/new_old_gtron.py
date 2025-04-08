import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("../pythontools"))
import mmd_tools
import torch
from torch import nn
import numpy as np

V = "2"

#og_model = "UFNLP/gatortron-base"
#new_model = "fine_tuned_gatortron_V"+V
model = "sentence"

df = pd.read_csv("../../dev-set-alpha.csv")

og_embeddings = mmd_tools.get_doc_embeddings(list(df['ORIGINAL']), model_name = model)
#og_embeddings_new_model = mmd_tools.get_doc_embeddings(list(df['ORIGINAL']), model_name = new_model)
reference_embeddings = mmd_tools.get_doc_embeddings(list(df['REFERENCE']), model_name = model)
#reference_embeddings_new_model = mmd_tools.get_doc_embeddings(list(df['REFERENCE']), model_name = new_model)

np.save(f'../../og_embeddings_sentence_V{V}.npy', og_embeddings)
#np.save(f'../../og_embeddings_new_model_V{V}.npy', og_embeddings_new_model)
np.save(f'../../reference_embeddings_sentence_V{V}.npy', reference_embeddings)
#np.save(f'../../reference_embeddings_new_model_V{V}.npy', reference_embeddings_new_model)

