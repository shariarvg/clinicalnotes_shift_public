import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_lambda=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        # x shape: (batch_size * seq_len, hidden_dim)
        encoded = torch.sigmoid(self.encoder(x))
        decoded = self.decoder(encoded)
        
        # Sparsity penalty: encourage activations close to zero
        sparsity_penalty = self.sparsity_lambda * torch.mean(torch.abs(encoded))
        return decoded, encoded, sparsity_penalty
