import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
ep = MIMICEndpoint()
import numpy as np
import mmd_tools
import torch

V = 0

with open("../../api_key_gemini.txt", 'r') as f:
    api_key = f.readline()
    
import google.generativeai as genai


genai.configure(api_key=api_key)

# Select the Gemini model you want to use
model = genai.GenerativeModel("gemini-1.5-flash")

def get_summary(note, model = model): 
    start = "You are a helpful medical assistant whose job is to make clinical notes more readable. Your goal is to remove clinical jargon and template-written portions of the notes.Rewrite this note so that it is more understandable: "
    return model.generate_content(start + note).text

def get_summaries(notes, model = model):
    return [get_summary(note, model) for note in notes]

ms = MIMICSource(ep, "get_mixture", ["Z515", "Z66", "N170"], [10, 10, 10], [0.4, 0.4, 0.2])

notes_df = ms.obtain_samples(30)
note_ids = np.array(notes_df['note_id'])
notes = list(notes_df['text'])
summaries = get_summaries(notes)
emb_notes = mmd_tools.get_doc_embeddings(notes)
emb_summaries = mmd_tools.get_doc_embeddings(summaries)
dists = emb_notes - emb_summaries
dists_rand = emb_notes - emb_summaries[torch.randperm(emb_summaries.shape[0])]
dists2 = (dists*dists).sum(axis = 1)
dists_rand2 = (dists_rand*dists_rand).sum(axis = 1)

torch.save(emb_notes, f'../../note_embeddings_V{V}.pt')
torch.save(emb_summaries, f'../../summary_embeddingsV{V}.pt')
torch.save(dists2, f'../../dists2_V{V}.pt')
torch.save(dists_rand2, f'../../dists_rand2_V{V}.pt')
np.save(f'../../note_id_V{V}.pt', note_ids)

