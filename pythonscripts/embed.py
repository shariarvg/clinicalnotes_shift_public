'''
Short script that I used to obtain some embeddings, just to look at them
'''

import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath("../pythontools"))
import mmd_tools
from mimic_tools import MIMICEndpoint
from sklearn.decomposition import PCA
import sys

ep = MIMICEndpoint()

V = 9

#codes10 = ["F411", "F329", "F4310", "F419", "F339", "F422", "F902"]
#codes9 = ["30002", "29620", "F4310", "30000", "29630", "3071", "31401"]
#long_titles = ["Generalized anxiety disorder", "Major depressive disorder, single episode, unspecified", "Post-traumatic stress disorder, unspecified", "Anxiety disorder, unspecified", "Major depressive disorder, recurrent, unspecified", "Anorexia nerviosa", "attention-deficit hyperactivity disorder, combined type"]
'''
all_notes = pd.DataFrame()
for code in codes9:
    new_notes = ep.get_notes_diagnosis(code, 9)
    new_notes['code'] = [code for count in range(new_notes.shape[0])]
    new_notes['vers'] = [code for count in range(new_notes.shape[0])]
    all_notes = pd.concat([all_notes, new_notes.sample(min(500, new_notes.shape[0]))])
for code in codes10:
    new_notes = ep.get_notes_diagnosis(code, 10)
    new_notes['code'] = [code for count in range(new_notes.shape[0])]
    new_notes['vers'] = [code for count in range(new_notes.shape[0])]
    all_notes = pd.concat([all_notes, new_notes.sample(min(500, new_notes.shape[0]))])
'''   

'''
CODE = str(sys.argv[1])
all_notes = ep.get_notes_diagnosis(CODE, 10)

embeddings = mmd_tools.get_doc_embeddings(list(all_notes['text']))

df = pd.DataFrame(embeddings, columns = [f"dim{x}" for x in range(embeddings.shape[1])])
df['start_year'] = list(all_notes['start_year'])
df.to_csv(f"embeddings_{CODE}.csv")
'''

'''
V = 6

notes = pd.read_csv("../../v1_jargon_pairs.csv")

base_emb_note_with_jargon = mmd_tools.get_doc_embeddings(list(notes['sentence_with_jargon']), model_name = "UFNLP/gatortron-base", summary = "mean")
base_emb_note_without_jargon = mmd_tools.get_doc_embeddings(list(notes['sentence_without_jargon']), model_name = "UFNLP/gatortron-base", summary = "mean")
ft_emb_note_with_jargon = mmd_tools.get_doc_embeddings(list(notes['sentence_with_jargon']), model_name = "fine_tuned_gatortron_V2", summary = "mean")
ft_emb_note_without_jargon = mmd_tools.get_doc_embeddings(list(notes['sentence_without_jargon']), model_name = "fine_tuned_gatortron_V2", summary = "mean")


np.save(f"../../base_emb_note_with_jargon_V{V}.npy", base_emb_note_with_jargon)
np.save(f"../../base_emb_note_without_jargon_V{V}.npy", base_emb_note_without_jargon)
np.save(f"../../ft_emb_note_with_jargon_V{V}.npy", ft_emb_note_with_jargon)
np.save(f"../../ft_emb_note_without_jargon_V{V}.npy", ft_emb_note_without_jargon)
'''

codes9 = ["4019", "29620", "V667", "42833", "27801"]
codes10 = ["I10", "F329", "Z515", "I5033", "E6601"]


MAX_SAMPLE = 2000

for code in codes9:
    notes = ep.get_notes_diagnosis(code, 9)
    notes = notes.sample(min(notes.shape[0], MAX_SAMPLE))
    emb_b_mean = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = "UFNLP/gatortron-base", max_length = 300)
    emb_b_first = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = "UFNLP/gatortron-base", max_length = 300, summary = "first")
    emb_f_mean = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = "fine_tuned_gatortron_V2", max_length = 300)
    emb_f_first = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = "fine_tuned_gatortron_V2", max_length = 300, summary = "first")
    
    np.save(f"../../embeddings_{code}_9_gatortronbase_mean.npy", emb_b_mean)
    np.save(f"../../embeddings_{code}_9_gatortronbase_first.npy", emb_b_first)
    np.save(f"../../embeddings_{code}_9_gatortronFTV2_mean.npy", emb_f_mean)
    np.save(f"../../embeddings_{code}_9_gatortronFTV2_first.npy", emb_f_first)
    
for code in codes10:
    notes = ep.get_notes_diagnosis(code, 10)
    notes = notes.sample(min(notes.shape[0], MAX_SAMPLE))
    emb_b_mean = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = "UFNLP/gatortron-base", max_length = 300)
    emb_b_first = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = "UFNLP/gatortron-base", max_length = 300, summary = "first")
    emb_f_mean = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = "fine_tuned_gatortron_V2", max_length = 300)
    emb_f_first = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = "fine_tuned_gatortron_V2", max_length = 300, summary = "first")
    
    np.save(f"../../embeddings_{code}_10_gatortronbase_mean.npy", emb_b_mean)
    np.save(f"../../embeddings_{code}_10_gatortronbase_first.npy", emb_b_first)
    np.save(f"../../embeddings_{code}_10_gatortronFTV2_mean.npy", emb_f_mean)
    np.save(f"../../embeddings_{code}_10_gatortronFTV2_first.npy", emb_f_first)


notes = ep.get_notes_version(version = 10, total_size = 2000)
emb_b_mean = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = "UFNLP/gatortron-base", max_length = 300)
emb_b_first = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = "UFNLP/gatortron-base", max_length = 300, summary = "first")
emb_f_mean = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = "fine_tuned_gatortron_V2", max_length = 300)
emb_f_first = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = "fine_tuned_gatortron_V2", max_length = 300, summary = "first")
    
code = "ALL"
np.save(f"../../embeddings_{code}_10_gatortronbase_mean.npy", emb_b_mean)
np.save(f"../../embeddings_{code}_10_gatortronbase_first.npy", emb_b_first)
np.save(f"../../embeddings_{code}_10_gatortronFTV2_mean.npy", emb_f_mean)
np.save(f"../../embeddings_{code}_10_gatortronFTV2_first.npy", emb_f_first)


print(f"File {os.path.basename(__file__)} V{V}")