import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
commit_hash = sys.argv[1]
commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash


disease = sys.argv[2]
year1 = int(sys.argv[3])
year2 = int(sys.argv[4])
mbn_to_extended_name = {"gpt2": "gpt2", "gtron": "UFNLP/gatortron-base"}


for model_name in ['gpt2', 'gtron', 'bow']:
    if "bow" not in model_name:
        tokenizer = AutoTokenizer.from_pretrained(mbn_to_extended_name[model_name])
    if "gpt2" in model_name:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    max_length = 100
    n_epochs = 100




    device = 'cuda'


    class NotesDataset(torch.utils.data.Dataset):
        def __init__(self, data, task):
            self.notes_data = list(data['text']) # List of tuples (original, positive, negative)
            self.labels = list(data[task])

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

    class ModelWithClassifier(nn.Module):
        def __init__(self, base_model, num_labels):
            super(ModelWithClassifier, self).__init__()
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

    def fine_tune_task_dataset(notes_dataset, save_name):
        if "bow" in model_name.lower():
            train_bow_dataset(notes_train, save_name)
            return

        model_base = AutoModel.from_pretrained(mbn_to_extended_name[model_name]).to(device)

        if "gpt" in model_name:
            model_base.resize_token_embeddings(len(tokenizer))


        criterion = nn.CrossEntropyLoss()
        dataset = NotesDataset(notes_dataset, 'task')#.to(device)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        model = ModelWithClassifier(model_base, 2).to(device)
        model.load_state_dict(torch.load(f"../../experimentresults/{disease}_{year1}_{year2}_{model_name}_model.pt"))
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
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update learning rate
                lr_scheduler.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1} of {save_name} completed")

        torch.save(model.state_dict(), save_name + "_model.pt")

    def train_bow_dataset(notes_train, save_name):
        cv = CountVectorizer(min_df = 0.05, max_df = 0.95)
        emb_train = cv.fit_transform(list(notes_train['text'])).toarray()
        rfc = RandomForestClassifier(max_depth = 5)
        rfc.fit(emb_train, notes_train['task'])
        joblib.dump(cv, save_name + "_cv.pt")
        joblib.dump(rfc, save_name + "_rfc.pt")


    ep = MIMICEndpoint()


    save_name_notes = f'../../experimentresults/{disease}_{year1}_{year2}'
    if disease in ['admission_in_30_days', 'death']:
        note_ids = np.load(f'../../experimentresults/cardio_{year1}_{year2}' + "_note_id.npy", allow_pickle = True)
        notes = pd.DataFrame({"note_id": note_ids})
        notes = pd.merge(notes, ep.notes[['note_id', 'text', 'start_year', disease]], how = 'left', on = 'note_id') 
        notes['task'] = notes[disease].values
    else:
        note_ids = np.load(save_name_notes + "_note_id.npy", allow_pickle = True)
        answers = pd.read_csv(save_name_notes + "_diagnosis_answers.csv")
        answers['task'] = answers['0'].str.contains("Yes").astype(int)
        notes = pd.DataFrame({"note_id": note_ids})

        notes = pd.merge(notes, ep.notes[['note_id', 'text', 'start_year']], how = 'left', on = 'note_id')
        notes['task'] = answers['task'].values

    notes_train = notes[(notes['start_year']==year1)].iloc[:5000]

    ## ensure training set is balanced!!!!
    notes_train_pos = notes_train[(notes_train['task']==1)].iloc[:750]
    notes_train_neg = notes_train[(notes_train['task']==0)].iloc[:750]
    notes_train = pd.concat([notes_train_pos, notes_train_neg])

    save_name_model = f'../../experimentresults/{disease}_{year1}_{year2}_{model_name}'

    fine_tune_task_dataset(notes_train, save_name_model)

    with open(save_name_model + "_model_training.txt", 'w') as f:
        f.write("train_gen.py\n")
        f.write(commit_link)