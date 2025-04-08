import numpy as np
import sys, os
import pandas as pd
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
import mmd_tools
from mimic_source import MIMICSource
import torch
import torch.nn as nn

ep = MIMICEndpoint()
ms = MIMICSource(ep, "get_mixture", ["Z515", "Z66", "N170"], [10, 10, 10], [0.4, 0.4, 0.2])
model_name = "UFNLP/gatortron-base"

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()  # Encoded representation
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Reconstruct to match the input
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction
    
def gen_embeddings_and_noised_embeddings(notes, instruction1, instruction2):
    noised_notes = ep.generate_newtextcolumn_chatgpt(instruction1, instruction2, 'gpt-4', notes)
    embeddings = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = "UFNLP/gatortron-base")
    noised_embeddings = mmd_tools.get_doc_embeddings(list(noised_notes['text']), model_name = "UFNLP/gatortron-base")
    return embeddings, noised_embeddings

def train_denoising_autoencoder(autoencoder, instruction1, instruction2, notes, epochs = 10, batch_size = 16, loss = nn.MSELoss()):
    embeddings, noised_embeddings = gen_embeddings_and_noised_embeddings(notes, instruction1, instruction2)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr = 0.001)
    for i in range(epochs):
        epoch_loss = 0.0
        for i in range(0, notes.shape[0], batch_size):
            optimizer.zero_grad()
            embeddings_batch = embeddings[i:i+batch_size]
            noised_embeddings_batch = noised_embeddings[i:i+batch_size]
            encodings, reconstructions = autoencoder(noised_embeddings_batch)
            l = loss(reconstructions, embeddings)
            epoch_loss += l.item()
            l.backward()
            optimizer.step()
        print(f"Epoch {i} Loss: {epoch_loss}")

autoencoder = Autoencoder(1024, 128)
notes = ms.obtain_samples(100)

instruction1 = "You are a helpful medical assistant whose job is to make clinical notes more readable. Your goal is to remove clinical jargon and template-written portions of the notes."
instruction2 = "Rewrite this note so that it is more understandable: "
train_denoising_autoencoder(autoencoder, instruction1, instruction2, notes)