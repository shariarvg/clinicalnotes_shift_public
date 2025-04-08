'''
Training a Sparse Autoencoder on MIMIC Fine-tuned gatortron embeddings.
'''
import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from safetensors.torch import load_file

import torch
import torch.nn as nn
import torch.optim as optim
from mimic_tools import MIMICEndpoint
from featurization_tools import *
from mimic_source import MIMICSource

from sae import SparseAutoencoder
ep = MIMICEndpoint()

N_EPOCHS = 100
BATCH_SIZE = 8
HIDDEN_DIM = 16
SPARSITY_LAMBDA = 0.001
LEARNING_RATE = 0.001
criterion = nn.MSELoss()
N_NOTES = 1000
V = "topk_1"
K = 4

featurizer = TaskTunedTransformer(classifier_path = "../../gtron_death30")#/model.safetensors")#BOW()
fname = f"gtron_death_in_30_{HIDDEN_DIM}_{V}"
save_name = f"../../sae_{fname}.pth"

def get_data():
    # Load the MIMIC data
    ms1 = MIMICSource(ep, "get_notes_diagnosis", "Z515", 10)
    ms2 = MIMICSource(ep, "get_notes_diagnosis", "N170", 10)
    
    notes = pd.concat([ms1.obtain_samples(TOTAL_SIZE = int(N_NOTES/2)), ms2.obtain_samples(TOTAL_SIZE = int(N_NOTES/2))])
    embeddings = featurizer.transform(notes['text'])
    print("Embeddings shape: ", embeddings.shape)
    return embeddings

def train_sae(embeddings):
    model = SparseAutoencoder(input_dim = embeddings.shape[1], hidden_dim = HIDDEN_DIM, sparsity_lambda = SPARSITY_LAMBDA, topk = K)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    for epoch in range(N_EPOCHS):
        epoch_mse_loss = 0.0
        epoch_sparsity_loss = 0.0
        for i in range(0, len(embeddings), BATCH_SIZE):
            batch_embeddings = embeddings[i:i+BATCH_SIZE]
            batch_embeddings = torch.tensor(batch_embeddings, dtype=torch.float32)
            optimizer.zero_grad()
            d, e, p = model(batch_embeddings)
            mse_loss = criterion(d, batch_embeddings)
            sparsity_loss = p
            loss = mse_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            epoch_mse_loss += mse_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            
        print(f"Epoch {epoch+1} MSE Loss: {epoch_mse_loss/len(embeddings)}, Sparsity Loss: {epoch_sparsity_loss/len(embeddings)}")
        
    torch.save(model.state_dict(), save_name)
    return model

def main():
    embeddings = get_data()
    train_sae(embeddings)
    print(f"Model saved to {save_name}")
    
main()
