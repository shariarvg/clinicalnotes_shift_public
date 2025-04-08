import mauve
import torch
import time
from mimic_tools import MIMICEndpoint
ep = MIMICEndpoint()

notes1 = ep.get_notes_diagnosis("4019", 9)
notes2 = ep.get_notes_diagnosis("I10", 10)
notes3 = ep.get_notes_diagnosis("I5033", 10)

start = time.time()
print(mauve.compute_mauve(p_text = notes1['text'].sample(100), q_text = notes3['text'].sample(100), featurize_model_name = "gpt2"))
e1 = time.time()
print(e1 - start)
print("---")
print(mauve.compute_mauve(p_text = notes1['text'].sample(100), q_text = notes1['text'].sample(100), featurize_model_name = "gpt2"))
e2 = time.time()
print(e2 - e1)
print(mauve.compute_mauve(p_text = notes1['text'].sample(100), q_text = notes2['text'].sample(100), featurize_model_name = "jinaai/jina-embeddings-v2-small-en"))
e3 = time.time()
print(e3 - e2)
print(mauve.compute_mauve(p_text = notes1['text'].sample(100), q_text = notes3['text'].sample(100), featurize_model_name = "jinaai/jina-embeddings-v2-small-en"))
e4 = time.time()
print(e4 - e3)
print(mauve.compute_mauve(p_text = notes1['text'].sample(100), q_text = notes1['text'].sample(100), featurize_model_name = "jinaai/jina-embeddings-v2-small-en"))
e5 = time.time()
print(e5 - e4)