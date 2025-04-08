'''
Save sparse encodings and binary features for 1000 notes.
'''
import sys, os
sys.path.append(os.path.abspath("../pythontools"))

import torch
import numpy as np
import pandas as pd
from featurization_tools import TaskTunedTransformer
from mimic_tools import MIMICEndpoint
from sae import SparseAutoencoder

def load_sae_model(model_path, input_dim, hidden_dim):
    """Load the trained SAE model."""
    model = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_sparse_encodings(model, embeddings, batch_size=32):
    """Get sparse encodings for a batch of embeddings."""
    sparse_encodings = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32).to('cuda')
            _, encoded, _ = model(batch)
            sparse_encodings.append(encoded.cpu().numpy())
    return np.concatenate(sparse_encodings, axis=0)

def main():
    
    hidden_dim = 16
    V="topk_1"
    altV = None

    # Initialize MIMIC endpoint and get notes
    ep = MIMICEndpoint(root = "../../..", path = "../../../notes_preproc.csv")
    notes = pd.concat([ep.get_notes_diagnosis("Z515", 10, total_size=500), ep.get_notes_diagnosis("N170", 10, total_size=500)])
    #notes = pd.concat([ep.get_notes_death(1, total_size = 500), ep.get_notes_death(0, total_size = 500)])
    
    
    sae_path = f"../../sae_gtron_death_in_30_{hidden_dim}_{V}.pth"
    
    if altV is not None:
        V = str(V)+str(altV)
    
    # Get embeddings from death classifier
    featurizer = TaskTunedTransformer(classifier_path="../../gtron_death30")
    embeddings = featurizer.transform(notes['text'])
    
    # Load and run SAE
    input_dim = embeddings.shape[1]
    model = load_sae_model(sae_path, input_dim, hidden_dim)
    sparse_encodings = get_sparse_encodings(model, embeddings)
    
    # Save sparse encodings
    np.save(f'../../sparse_encodings_{hidden_dim}_{V}.npy', sparse_encodings)
    
    # Get binary features and note IDs
    binary_features = notes[['note_id', 'over_65', 'has_Z66', 'has_Z515', 'has_N170', 'gender', 'has_a419', 'ER', 'AST']]
    
    # Save binary features with note IDs
    binary_features.to_csv(f'../../binary_features_{hidden_dim}_{V}.csv', index=False)
    
    print(f"Sparse encodings shape: {sparse_encodings.shape}")
    print(f"Binary features shape: {binary_features.shape}")
    print(f"Files saved as:")
    print(f"- sparse_encodings_{hidden_dim}_{V}.npy")
    print(f"- binary_features_{hidden_dim}_{V}.csv")

if __name__ == "__main__":
    main() 