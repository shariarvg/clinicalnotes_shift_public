import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("../pythontools"))
import mmd_tools
import torch
from torch import nn
import numpy as np

V = "0"
df = pd.read_csv(f'note_ids_pre_translated_notes_V{V}.csv')

notes = pd.DataFrame()
for count in range(0, 1000, 200):
    notes_new = pd.read_csv(f"translated_notes_{str(count)}_{str(count+200)}_V{V}.csv")
    notes = pd.concat([notes, notes_new])
    
old_embeddings= mmd_tools.get_doc_embeddings(list(notes['OG']))
new_embeddings= mmd_tools.get_doc_embeddings(list(notes['New']))
'''
Adam optimization
class Transform(nn.Module):
    def __init__(self):
        self.nnLinear = nn.Linear(new_embeddings.shape[1], new_embeddings.shape[1])
        
    def forward(self, embedding):
        return self.nnLinear(embedding)
    
transform = Transform()

optimizer = torch.optim.Adam(transform.parameters(), lr = 0.001)

loss = nn.MSELoss()
batch_size = 25
for epoch in range(10):
    epoch_loss = 0.0
    for i in range(0, 1000, batch_size):
        optimizer.zero_grad()
        ie = old_embeddings[i:i+batch_size]
        oe = new_embeddings[i:i+batch_size]
        pred = transform(ie)
        l = loss(pred, oe)
        l.backward()
        optimizer.step()
        
        epoch+loss += l.item()
        
    print(f"Epoch {epoch} loss: ", epoch_loss/1000)
    

torch.save(transform.state_dict(), f'transform_V{V}.pt')
'''

n, d, k = 100, old_embeddings.shape[1], new_embeddings.shape[1]  # number of observations, input dim, output dim
lambda_reg = 1.0
I = np.eye(d)  # Identity matrix of size d x d
XtX = old_embeddings.T @ old_embeddings  # Compute X^T X
XtY = old_embeddings.T @ new_embeddings
W = np.linalg.solve(XtX + lambda_reg * I, XtY)

np.save(f'../../transformation_{V}.npy', W)
                                        
                          

                                              
                                    