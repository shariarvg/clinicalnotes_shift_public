from imports import *
import mmd_tools
import pickle
import sys
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

ep = MIMICEndpoint()

CODE1 = str(sys.argv[1])
VERS1 = int(sys.argv[2])
filename = str(sys.argv[3])

N_CLUSTERS = 8

notes = ep.get_notes_diagnosis(CODE1, VERS1)
notes = notes[(notes['text'].str.len() > 10)] ## discard '' documents

notes1 = notes[(notes['start_year'] == 2015)]
#notes2 = notes[(notes['start_year'] == 2017)]

e = mmd_tools.get_doc_embeddings(list(notes1['text']), model_name = 'gpt2-large')
km = KMeans(n_clusters = N_CLUSTERS).fit(e)
notes1['cluster'] = km.predict(e)

vectorizer = CountVectorizer(stop_words='english', min_df = 0.10, max_df = 0.97)  # Adjust stopwords as needed
X = vectorizer.fit_transform(notes1['text'])
words = vectorizer.get_feature_names_out()

word_counts_corpus = X.toarray().sum(axis=0)
corpus_counts_df = pd.DataFrame({'word': words, 'count': word_counts_corpus})

# Get total words in corpus
total_words_corpus = word_counts_corpus.sum()

def calc_meaningfulness(cluster):
    cluster_notes = notes1[(notes1['cluster']==cluster)]['text']
    cluster_X = vectorizer.transform(cluster_notes)
    cluster_word_counts = cluster_X.toarray().sum(axis=0)
    total_words_cluster = cluster_word_counts.sum()
    
    # Create a DataFrame for word counts in the cluster
    cluster_counts_df = pd.DataFrame({'word': words, 'count': cluster_word_counts})
    
    merged_df = pd.merge(cluster_counts_df, corpus_counts_df, on='word', suffixes=('_cluster', '_corpus'))
    
    # Calculate meaningfulness (frequency in cluster / frequency in corpus)
    merged_df['meaningfulness'] = (merged_df['count_cluster'] / total_words_cluster) / (merged_df['count_corpus'] / total_words_corpus)
    
    # Sort by meaningfulness and select top 3 words
    top_words = merged_df.sort_values(by='meaningfulness', ascending=False).dropna().head(10)
    return top_words[['word', 'meaningfulness']]

for cluster in range(N_CLUSTERS):
    print(cluster)
    print(calc_meaningfulness(cluster))
    
e = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = 'gpt2-large')
notes['cluster'] = km.predict(e)


with open(filename, 'wb') as out:
    pickle.dump(km, out)
    
notes[['hadm_id', 'cluster']].to_csv(filename+"_cluster_assignments.csv")