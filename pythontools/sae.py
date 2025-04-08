import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_lambda=1e-3, topk=10):
        super().__init__()
        self.encoder_weight = nn.Linear(input_dim, hidden_dim)
        self.encoder_pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder_post_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder_weight = nn.Linear(hidden_dim, input_dim)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.sparsity_lambda = sparsity_lambda
        self.topk = topk

    def forward(self, x):
        # x shape: (batch_size * seq_len, hidden_dim)
        pre_encoded = self.encoder_weight(x - self.encoder_pre_bias) + self.encoder_post_bias
        topk_values, topk_indices = torch.topk(pre_encoded, k=self.topk, dim=1)
        encoded = torch.zeros_like(pre_encoded)
        encoded.scatter_(1, topk_indices, topk_values)
        encoded = torch.relu(encoded)
        decoded = self.decoder_weight(encoded) + self.decoder_bias
        
        # Sparsity penalty: encourage activations close to zero
        sparsity_penalty = self.sparsity_lambda * torch.mean(torch.abs(encoded))
        return decoded, encoded, sparsity_penalty
