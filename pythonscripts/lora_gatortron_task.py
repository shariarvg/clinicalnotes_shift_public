import sys, os
import torch
sys.path.append(os.path.abspath("../pythontools"))
import pandas as pd
import numpy as np
from mimic_tools import MIMICEndpoint
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from notes_dataset import NotesDataset
from model_with_classifier import ModelWithClassifier
from mimic_source import MIMICSource, MIMICMultiSource
from influence_tools import compute_influence

save_name = "../../gatortron_readmission_prediction_lowrank"


'''
ep = MIMICEndpoint()
ms_young = MIMICSource(ep, "get_notes_key_value", "start_age", 100, False, True)
ms_2a = MIMICSource(ep, "get_notes_key_value", "start_age", 99, True, False)
ms_2b = MIMICSource(ep, "get_notes_key_value", "death_in_30_days", 1, False, False)
ms_2c = MIMICSource(ep, "get_notes_key_value", "death_in_30_days", 0, False, False)
ms_old_dead = MIMICMultiSource([ms_2a, ms_2b])
ms_old_survive = MIMICMultiSource([ms_2a, ms_2c])

notes_train = pd.concat([ms_young.obtain_samples(1000), ms_old_dead.obtain_samples(7), ms_old_survive.obtain_samples(1)])
notes_test = pd.concat([ms_young.obtain_samples(100), ms_old_dead.obtain_samples(1), ms_old_survive.obtain_samples(20)])
'''

task = "admission_in_30_days"

ep = MIMICEndpoint()
ms_young = MIMICSource(ep, "get_notes_key_value", "start_age", 70, False, True)
ms_2a = MIMICSource(ep, "get_notes_key_value", "start_age", 69, True, False)
ms_2b = MIMICSource(ep, "get_notes_key_value", task, 1, False, False)
ms_2c = MIMICSource(ep, "get_notes_key_value",task, 0, False, False)
ms_old_read = MIMICMultiSource([ms_2a, ms_2b])
ms_old_nonread = MIMICMultiSource([ms_2a, ms_2c])

notes_train = pd.concat([ms_young.obtain_samples(1000), ms_old_read.obtain_samples(50), ms_old_nonread.obtain_samples(5)])
notes_test = pd.concat([ms_young.obtain_samples(100), ms_old_read.obtain_samples(25), ms_old_nonread.obtain_samples(25)])

## LORA Config

config = LoraConfig(
    r=8,  # Rank of LoRA adaptation
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout probability
    bias="none",
    target_modules=["query", "key", "value"],  # Specify transformer layers to adapt
    task_type="SEQ_CLS"  # Task type (e.g., sequence classification)
)

tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base")
model = ModelWithClassifier(AutoModel.from_pretrained("UFNLP/gatortron-base"),2)

## Load model and get peft-trainable model
model = get_peft_model(model, config)
model.print_trainable_parameters()

## set training arguments
args = TrainingArguments(save_name, label_names = [task], num_train_epochs = 20)
dataset_train = NotesDataset(notes_train, task, tokenizer = tokenizer, max_length = 100)
dataset_eval = NotesDataset(notes_test.iloc[100:], task, tokenizer = tokenizer, max_length = 100)
trainer = Trainer(model, args, train_dataset = dataset_train, eval_dataset= dataset_eval)
trainer.train()

batch_size = 1

eval_results = trainer.evaluate()
print(eval_results)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
eval_loader = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)

np.save(save_name+"_influence.npy", compute_influence(model, train_loader, eval_loader, torch.nn.CrossEntropyLoss()))
np.save(save_name+"_note_id.npy", np.hstack([notes_train['note_id'], notes_test['note_id']]))
