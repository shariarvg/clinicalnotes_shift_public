import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import sys

commit_hash = sys.argv[1]

V_last = ""
V = "3"
V_trainset = "0"

model_name = "UFNLP/gatortron-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModel.from_pretrained("../../fine_tuned_gatortron_V"+V_last)
model = AutoModel.from_pretrained(model_name)

temperature = 0.05
cosine_similarity = nn.CosineSimilarity(dim=-1)
loss_fn = nn.CrossEntropyLoss()


class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data  # List of tuples (original, positive, negative)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
## IGNORE !
def contrastive_loss_old(embeddings, temperature):
    '''
    gpt written (wrong i think)
    '''
    batch_size = embeddings.size(0) // 3  # Assuming triplets: anchor, positive, negative
    anchor, positive, negative = (
        embeddings[:batch_size],
        embeddings[batch_size:2 * batch_size],
        embeddings[2 * batch_size:],
    )

    similarities = torch.cat(
        [cosine_similarity(anchor, positive).unsqueeze(1),
         cosine_similarity(anchor.unsqueeze(1), negative.unsqueeze(0))], dim=1
    )

    labels = torch.zeros(batch_size, dtype=torch.long, device=embeddings.device)
    return loss_fn(similarities / temperature, labels)

def contrastive_loss(embeddings, temperature, hard_negatives = True):
    '''
    i wrote
    '''
    batch_size = embeddings.size(0) // 3
    anchor, positive, negative = (
        embeddings[:batch_size]/embeddings[:batch_size].norm(dim=1,keepdim=True),
        embeddings[batch_size:2 * batch_size]/embeddings[batch_size:2 * batch_size].norm(dim=1,keepdim=True),
        embeddings[2 * batch_size:]/embeddings[2 * batch_size:].norm(dim=1,keepdim=True),
    )
    Ap = (1/temperature)* (anchor @ positive.T) ## ijth element is the cosine similarity between anchor i and positive j
    An = (1/temperature)* (anchor @ negative.T) ## ijth element is the cosine similarity between anchor i and positive j
    expAp = torch.exp(Ap)
    expAn = torch.exp(An)
    numerator = expAp.diagonal() # shape (batch_size,)
    if hard_negatives:
        denominator = (expAp + expAn).sum(dim = 1) # shape (batch_size,)

        return -torch.log(numerator/denominator).sum()
    denominator = (expAp).sum(dim = 1)
    return -torch.log(numerator/denominator).sum()

def train(model, dataloader, optimizer, device, epochs = 20):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            texts = [text for triplet in batch for text in triplet]
            inputs = tokenizer(texts, return_tensors='pt', padding = 'max_length', max_length = 100, truncation = True).to(device)

            outputs = model(**inputs)

            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token for simplicity
            loss = contrastive_loss(embeddings, temperature, hard_negatives = False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} Loss: {loss.item()}")
        
# Data preparation
df = pd.read_csv(f"../../training_simcse_V{V_trainset}.csv")[['Anchor', 'Positive', 'HardNegative']]
data = [tuple(row) for row in df.itertuples(index=False, name=None)]

##
dataset = ContrastiveDataset(data)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)

# Train the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
train(model, dataloader, optimizer, device)

# Define save directory
save_directory = "./fine_tuned_gatortron_V"+V

# Save the model and tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")

commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash

with open(f"gatortron_V{V}.txt", 'w') as f:
    f.write("Made with train_simcse.py\n")
    f.write(commit_link)