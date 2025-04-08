import sys, os
sys.path.append(os.path.abspath("../pythontools"))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from featurization_tools import TaskTunedTransformer, BOW
from mimic_tools import MIMICEndpoint
from sae import SparseAutoencoder
from mimic_source import MIMICSource

'''
gender_dim = 544
over_65_dim = 706
z66_dim = 791
er_dim = 514
z515_dim = 818
'''
gender_dim = 749
#z515_dim = 818
#a419_dim = 433
#z66_dim = 767
n170_dim = 917

#dims = [gender_dim, over_65_dim, z66_dim, er_dim, z515_dim]
#dims = [gender_dim, z515_dim, a419_dim, z66_dim, n170_dim]
dims = [gender_dim, n170_dim]
ep = MIMICEndpoint()
source1 = MIMICSource(ep, "get_notes_key_equals_value", "gender", 1)
source2 = MIMICSource(ep, "get_notes_start_age_greaterthan", 65)
source3 = MIMICSource(ep, "get_notes_key_equals_value", "ER", 1)
source4 = MIMICSource(ep, "get_notes_diagnosis", "Z515", 10)
source5 = MIMICSource(ep, "get_notes_diagnosis", "Z66", 10)
source6 = MIMICSource(ep, "get_notes_diagnosis", "A419", 10)
source7 = MIMICSource(ep, "get_notes_diagnosis", "N170", 10)
sources = [source1, source2, source3, source4, source5, source6, source7]

notes = [source.obtain_samples(TOTAL_SIZE = 1000) for source in sources]

featurizer = TaskTunedTransformer(classifier_path = "../../gtron_death30")

embeddings = [featurizer.transform(note['text']) for note in notes]

sae = SparseAutoencoder(input_dim = embeddings[0].shape[1], hidden_dim = 1000)
sae.load_state_dict(torch.load("../../sae_gtron_death_in_30_1000.pth"))
sae.eval()

# Process embeddings in batches
batch_size = 32
sparse_encodings = []
for embedding in embeddings:
    batch_encodings = []
    with torch.no_grad():
        for i in range(0, len(embedding), batch_size):
            batch = torch.tensor(embedding[i:i+batch_size], dtype=torch.float32)
            _, encoded, _ = sae(batch)
            batch_encodings.append(encoded.numpy())
    sparse_encodings.append(np.concatenate(batch_encodings, axis=0))

for i in range(len(sparse_encodings)):
    for d in dims:
        print(f"Source {i+1}, Dimension {d}: {sparse_encodings[i][:, d].mean():.4f}")















