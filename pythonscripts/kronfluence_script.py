import kronfluence
import torch
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from model_with_classifier import ModelWithClassifier
from peft import PeftModel
from notes_dataset import NotesDataset
from transformers import AutoModel, AutoTokenizer
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence_task import CrossEntropyTask
print('imports successful')
ep = MIMICEndpoint()
from safetensors.torch import safe_open

task_name = "death_in_30_days"
save_name = "../../gatortron_"+task_name

checkpoint = 2500
path = save_name + f"/checkpoint-{checkpoint}"


note_ids_train = pd.read_csv(save_name + "_note_ids_train.csv")
note_ids_test = pd.read_csv(save_name + "_note_ids_test.csv")
note_train = pd.merge(note_ids_train, ep.notes, on = 'note_id', how = 'left')
note_test = pd.merge(note_ids_test, ep.notes, on = 'note_id', how = 'left')
#notes = pd.merge(pd.concat([pd.DataFrame({"note_id": note_ids})]), ep.notes, on = "note_id", how = "left")


classifier = ModelWithClassifier("UFNLP/gatortron-base", 2)
with safe_open(path, framework="pt", device="cpu") as f:
    state_dict = {key: f.get_tensor(key) for key in f.keys()}
classifier.load_state_dict(state_dict)
                  
        

# Before running Kronfluence
#for param in classifier.base_model.model.classifier.original_module.parameters():
#    param.requires_grad = True

train_dataset = NotesDataset(notes_train, task_name, AutoTokenizer.from_pretrained("UFNLP/gatortron-base"), 100)#.to('cuda')
eval_dataset = NotesDataset(notes_eval, task_name, AutoTokenizer.from_pretrained("UFNLP/gatortron-base"), 100)#.to('cuda')

task = CrossEntropyTask()


kron_model = prepare_model(
    model=classifier,
    task=task
)

analyzer = Analyzer(analysis_name = task_name, model = kron_model, task = task, output_dir = save_name + "_influence_results")

analyzer.fit_all_factors(factors_name=task_name+"_factors", dataset=train_dataset, per_device_batch_size = 4)

analyzer.compute_pairwise_scores(
    scores_name=task_name+"_scores",
    factors_name=task_name+"_factors",
    query_dataset=eval_dataset,
    train_dataset=train_dataset,
    per_device_query_batch_size=4,
)

scores = analyzer.load_pairwise_scores(scores_name=task_name+"_scores")

np.save(save_name+"_kronfluence_scores.npy", scores)