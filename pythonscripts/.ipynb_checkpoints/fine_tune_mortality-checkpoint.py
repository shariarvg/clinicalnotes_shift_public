import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource

from torch.optim import AdamW
from transformers import get_scheduler

ep = MIMICEndpoint()
notes = pd.concat([ep.get_notes_death(1, 1500), ep.get_notes_death(0, 1500)])

V = "2"

notes[['note_id']].to_csv(f"gatortron_death_classifer_note_ids_V{V}.csv")

n_epochs = 100

model_name = "UFNLP/gatortron-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_base = AutoModel.from_pretrained(model_name)

max_length = 100



class NotesDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.notes_data = list(data['text']) # List of tuples (original, positive, negative)
        self.labels = list(data['death'])
        
    def __len__(self):
        return len(self.notes_data)

    def __getitem__(self, idx):
        text = self.notes_data[idx]
        label = self.labels[idx]
        
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),  # Remove batch dimension
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

class GatorTronWithClassifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super(GatorTronWithClassifier, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)  # Regularization
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Pass inputs through the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Use the [CLS] token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Apply dropout and classifier
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

criterion = nn.CrossEntropyLoss()
dataset = NotesDataset(notes)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
model = GatorTronWithClassifier(model_base, 2)
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(dataloader) * n_epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

## NOT included in original V. meant to speed up training
for name, param in model_base.named_parameters():
    if not any(layer in name for layer in ["encoder.layer.21", "encoder.layer.22", "encoder.layer.23", "embeddings"]):
        param.requires_grad = False

for epoch in range(n_epochs):
    epoch_loss = 0.0
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate
        lr_scheduler.step()
        
        epoch_loss += loss.item()
    print(f"Epoch {epoch} loss: {epoch_loss/len(dataloader)}")
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"gatortron_death_classifier_chkpt_epoch{epoch}_V{V}.pt")
    
torch.save(model.state_dict(), f"gatortron_death_classifier_V{V}.pt")
                                 