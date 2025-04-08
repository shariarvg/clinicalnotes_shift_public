import torch
import sys, os
sys.path.append(os.path.abspath("../pythontools"))
import pandas as pd
import numpy as np
from mimic_tools import MIMICEndpoint
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from notes_dataset import NotesDataset
from model_with_classifier import ModelWithClassifier
from mimic_source import MIMICSource, MIMICMultiSource
from influence_tools import compute_influence

task = "death_in_30_days"

save_name = "../../gatortron_"+task

ep = MIMICEndpoint()

notes_death = ep.get_notes_key_value("death_in_30_days", 1, total_size = 500)
notes_alive = ep.get_notes_key_value("death_in_30_days", 0, total_size = 500)
notes_train = pd.concat([notes_death, notes_alive])
notes_test = pd.concat([notes_death.iloc[:50], notes_alive.iloc[:50]])

notes_train[['note_id']].to_csv(save_name+"_note_ids_train.csv")
notes_test[['note_id']].to_csv(save_name+"_note_ids_test.csv")


tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base")
model = ModelWithClassifier(AutoModel.from_pretrained("UFNLP/gatortron-base"),2)

args = TrainingArguments(save_name, label_names = [task], num_train_epochs = 100)
dataset_train = NotesDataset(notes_train, task, tokenizer = tokenizer, max_length = 100)
dataset_eval = NotesDataset(notes_test.iloc[100:], task, tokenizer = tokenizer, max_length = 100)
trainer = Trainer(model, args, train_dataset = dataset_train, eval_dataset= dataset_eval)
trainer.train()


