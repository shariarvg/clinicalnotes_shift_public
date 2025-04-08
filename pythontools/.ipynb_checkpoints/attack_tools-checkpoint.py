import pandas as pd
import numpy as np
import joblib
import re
import sys
import mmd_tools
import mimic_tools
from mimic_source import MIMICSource
from sklearn.feature_extraction.text import CountVectorizer
from abc import ABC, abstractmethod
import google.generativeai as genai


'''
Filtering attacks
'''

class FilteringAttack(ABC):
    
    @abstractmethod
    def get_notes_to_remove(self):
        pass
    def remove_ep(self):
        self.source.ep.delete_notes(self.notes_to_remove)
        
    def attack_source_ep(self):
        self.get_notes_to_remove()
        self.remove_ep()
        
    def add_back_removed_notes(self):
        self.source.ep.notes = pd.concat([self.source.ep.notes, self.notes_to_remove])
        self.notes_to_remove = None
        
    def get_attacked_sample(self, source, N, delete = False):
        self.attack_source_ep()
        sample = source.obtain_samples(N)
        if not delete:
            self.add_back_removed_notes()
        return sample
        
class TabularUnivariateFilteringAttack(FilteringAttack):
    def __init__(self, source, column, threshold=None, greaterthan = True, lb_filter = None, ub_filter = None):
        self.source = source
        self.column = column
        self.threshold = threshold
        self.greaterthan = greaterthan
        self.notes_to_remove = None
        self.lb_filter = lb_filter
        self.ub_filter = ub_filter
        
    def get_notes_to_remove(self):
        if self.greaterthan:
            self.notes_to_remove = self.source.ep.notes[(self.source.ep.notes[self.column] > self.threshold)]
        else:
            self.notes_to_remove = self.source.ep.notes[(self.source.ep.notes[self.column] < self.threshold)]
            
    def get_attacked_sample(self, source, N, delete = True):
        if self.threshold is None:
            assert ((self.lb_filter is not None) and (self.ub_filter is not None)), "Threshold or both bounds must be defined"
            
            self.threshold = np.random.randint(self.lb_filter, self.ub_filter)
            result = super().get_attacked_sample(source, N, delete)
            self.threshold = None
            return result
        return super().get_attacked_sample(source, N, delete)
            
                
class TabularBinaryFilteringAttack(FilteringAttack):
    def __init__(self, source, column, keep=0):
        self.source = source
        self.column = column
        self.keep = keep
        self.notes_to_remove = None
        
    def get_notes_to_remove(self):
        self.notes_to_remove = self.source.ep.notes[(self.source.ep.notes[self.column] == 1 - self.keep)]
            
    
class TabularMultivariateFilteringAttack(FilteringAttack):
    def __init__(self, source, columns, coefs, threshold, greaterthan = True):
        self.source = source
        self.columns = column
        self.coefs = coefs
        self.threshold = threshold
        self.greaterthan = greaterthan
        self.notes_to_remove = None
                                            
    def get_notes_to_remove(self):
        if self.greaterthan:
            self.notes_to_remove = self.source.ep.notes[(self.source.ep.notes[self.columns].values @ self.coefs > self.threshold)]
            
class KeywordFilteringAttack(FilteringAttack):
    def __init__(self, source, keyword):
        self.source = source
        self.keyword = keyword
        self.notes_to_remove = None
        
    def get_notes_to_remove(self):
        self.notes_to_remove = self.source.ep.get_notes_with_keyword(self.keyword, total_size = None)
            
        
'''
Perturbation attacks
'''
        
class PerturbationAttack(ABC):
    
    @abstractmethod
    def attack_note(self, note):
        pass
    
    def attack_sample(self, dataset_text):
        return dataset_text.apply(lambda x: self.attack_note(x))
    
    def get_attacked_sample(self, source, N, delete = False):
        sample = source.obtain_samples(N).copy()
        sample['text'] = self.attack_sample(sample['text'])
        return sample
    
class ProbabilisticRemoveFirstSentenceAttack(PerturbationAttack):
    def __init__(self, source, prob_drop = None):
        self.source = source
        self.prob = prob_drop
        
    def attack_note(self, note):
        if "." in note and np.random.uniform() < self.prob:
            return note[note.find(".")+1:]
        return note
    
    def get_attacked_sample(self, source, N, delete = False):
        if self.prob is None:
            self.prob = np.random.uniform()
            result = super().get_attacked_sample(source, N, delete)
            self.prob = None
            return result
        return super().get_attacked_sample(source, N, delete)
    
class ProbabilisticRemoveLastSentenceAttack(PerturbationAttack):
    def __init__(self, source, prob_drop = None):
        self.source = source
        self.prob = prob_drop
        
    def attack_note(self, note):
        if "." in note and np.random.uniform() < self.prob:
            lastp = note.rfind('.')
            if lastp + 5 < len(note): # there is another sentence without a period after this period
                return note[:lastp]
            stlastp = note[:lastp].rfind('.')
            return note[:stlastp]
        
    def get_attacked_sample(self, source, N, delete = False):
        if self.prob is None:
            self.prob = np.random.uniform()
            result = super().get_attacked_sample(source, N, delete)
            self.prob = None
            return result
        return super().get_attacked_sample(source, N, delete)
        
class ProbabilisticRemoveSentenceContainingAttack(PerturbationAttack):
    def __init__(self, source, key_words, prob_drop = None):
        self.source = source
        if isinstance(key_words, str):
            self.key_words = [key_words]
        else:
            self.key_words = key_words
        self.prob = prob_drop
        
    def attack_note_with_one_word(self, note, key_word):
        if key_word in note:
            f = note.find(key_word)
            p_before = max(0,note[:f].rfind("."))
            p_after = note[f:].find(".")
            return note[:p_before]+note[p_after:]
        return note
        
    def attack_note(self, note):
        if np.random.uniform() < prob_drop:
            for key_word in self.key_words:
                note = self.attack_note_with_one_word(note, key_word)
            
        return note
    
    def get_attacked_sample(self, source, N, delete = False):
        if self.prob is None:
            self.prob = np.random.uniform()
            result = super().get_attacked_sample(source, N, delete)
            self.prob = None
            return result
        return super().get_attacked_sample(source, N, delete)
    
class ProbabilisticOnlyIssuesAttack(PerturbationAttack):
    def __init__(self, source, prob_drop = None):
        self.source = source
        self.prob = prob_drop
        
    def attack_note(self, note):
        issues_markers = ['active issues', 'chronic issues', 'translational issues', 'chronic issues', 'stable issues']
        issues_start = [note.lower().find(marker) for marker in issues_markers]
        min_issues_start = min([float('inf') if iss == -1 else iss for iss in issues_start])
        if min_issues_start < float('inf') and np.random.uniform() < self.prob:
            return note[min_issues_start:]
        return note
        
    def get_attacked_sample(self, source, N, delete = False):
        if self.prob is None:
            self.prob = np.random.uniform()
            result = super().get_attacked_sample(source, N, delete)
            self.prob = None
            return result
        return super().get_attacked_sample(source, N, delete)

                
class ProbabilisticRemoveIssuesAttack(PerturbationAttack):
    def __init__(self, source, prob_drop = None):
        self.source = source
        self.prob = prob_drop
        
    def attack_note(self, note):
        issues_markers = ['active issues', 'chronic issues', 'translational issues', 'chronic issues', 'stable issues']
        issues_start = [note.lower().find(marker) for marker in issues_markers]
        min_issues_start = min([float('inf') if iss == -1 else iss for iss in issues_start])
        if min_issues_start < float('inf') and np.random.uniform() < self.prob:
            return note[:min_issues_start]
        return note
        
    def get_attacked_sample(self, source, N, delete = False):
        if self.prob is None:
            self.prob = np.random.uniform()
            result = super().get_attacked_sample(source, N, delete)
            self.prob = None
            return result
        return super().get_attacked_sample(source, N, delete)
    
            
class ProbabilisticSwitchToKeywordAttack(PerturbationAttack):
    def __init__(self, source, keyword, prob_switch = None):
        self.source = source
        self.keyword = keyword
        self.prob = prob_switch
        
    def attack_note(self, note):
        words = note.split()
        # Create a mask where each word has a 0.5 probability of being replaced
        mask = np.random.rand(len(words)) < self.prob
        attacked_words = [(self.keyword if m else w) for w, m in zip(words, mask)]
        attacked_note = " ".join(attacked_words)
        return attacked_note
    
    def get_attacked_sample(self, source, N, delete = False):
        if self.prob is None:
            self.prob = np.random.uniform()
            result = super().get_attacked_sample(source, N, delete)
            self.prob = None
            return result
        return super().get_attacked_sample(source, N, delete)
            
class ProbabilisticKeywordDropAttack(PerturbationAttack):
    def __init__(self, source, keyword, prob_drop = None):
        self.source = source
        self.keyword = keyword
        self.prob = prob_drop
        
    def attack_note(self, note):
        words = note.split()
        # Create a mask where each word has a 0.5 probability of being replaced
        mask = np.random.rand(len(words)) < self.prob
        attacked_words = [("" if (m and w==self.keyword) else w) for w, m in zip(words, mask)]
        attacked_note = " ".join(attacked_words)
        return attacked_note
    
    def get_attacked_sample(self, source, N, delete = False):
        if self.prob is None:
            self.prob = np.random.uniform()
            result = super().get_attacked_sample(source, N, delete)
            self.prob = None
            return result
        return super().get_attacked_sample(source, N, delete)
    
class ProbabilisticShortenAttack(PerturbationAttack):
    def __init__(self, source, max_len):
        self.source = source
        self.max_len = max_len
        
    def attack_note(self, note):
        words = note.split()
        if len(words) < self.max_len:
            return note
        return " ".join(words[:self.max_len])
    
    def get_attacked_sample(self, source, N, delete = False):
        if self.max_len is None:
            self.max_len = np.random.randint(5,50)
            result = super().get_attacked_sample(source, N, delete)
            self.max_len = None
            return result
        return super().get_attacked_sample(source, N, delete)
    
class ProbabilisticMultiplyFloatsAttack(PerturbationAttack):
    def __init__(self, source, factor = None, prob = 0.5):
        self.source = source
        self.factor = factor
        self.prob = prob
        
    def attack_note(self, note):
        def is_float(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
        words = note.split()
        attacked_words = [(str(float(s)*self.factor) if (is_float(s) and np.random.uniform() < self.prob) else s) for s in words]
        return " ".join(attacked_words)
    
    def get_attacked_sample(self, source, N, delete = False):
        if self.factor is None:
            self.factor = np.random.uniform(1,5)
            result = super().get_attacked_sample(source, N, delete)
            self.factor = None
            return result
        return super().get_attacked_sample(source, N, delete)
    
class ProbabilisticReplaceJargonWithExpansion(PerturbationAttack):
    def __init__(self, source, prob = None):
        self.source = source
        self.prob = prob
        df = pd.read_csv("../../question_jargon_keywords.csv")
        self.dic = df.set_index(df.columns[0])[df.columns[1]].to_dict()
        
    def attack_note(self, note):
        note_split = note.split()
        for i, word in enumerate(note_split):
            if word in self.dic.keys() and np.random.uniform() < self.prob:
                note_split[i] = self.dic[word]
        return " ".join(note_split)
    
class ProbabilisticReplaceWordWithWordAttack(PerturbationAttack):
    def __init__(self, source, old, new, prob = None):
        self.source = source
        self.old = old
        self.new = new
        self.prob = prob
        
    def attack_note(self, note):
        note_list = note.split()
        for i, word in enumerate(note_list):
            if word ==self.old and np.random.uniform() < self.prob:
                note_list[i] = self.new
        return " ".join(note_list)

    
class ProbabilisticReplaceWithNextAttack(PerturbationAttack):
    def __init__(self, source, prob = None):
        self.source = source
        self.prob = prob
        
    def attack_note(self, note):
        return
    
    def get_attacked_sample(self, source, N, delete = False):
        '''
        Unusual thing is that attack note just isn't implemented here
        '''
        sample = source.obtain_samples(N).copy()
        def get_next_note(subj_id, note_seq, ep):
            notes = ep.notes[((ep.notes['subject_id']==subj_id)&(ep.notes['note_seq']==note_seq))]
            if notes.shape[0]==0:
                return None
            return notes['text'].iloc[0]
        
        next_text = sample.apply(lambda x: get_next_note(x['subject_id'], x['note_seq']+1, source.ep), axis = 1)
        
        if self.prob is None:
            p = np.random.uniform()
        else:
            p = self.prob
        
        mask = (~next_text.isna()) & (np.random.uniform() < p)
        sample['text'] = np.where(mask, next_text, sample['text'])
        return sample
    
class ProbabilisticLabelBasedAdditionAttack(PerturbationAttack):
    def __init__(self, source, column, value, addition, prob = None):
        self.source = source
        self.column = column
        self.value = value
        self.addition = addition
        self.prob = prob
        
    def attack_note(self, note, value):
        if value == self.value and np.random.uniform() < self.prob:
            return self.addition + " " + note
        return note
    
    def attack_sample(self, dataset_text_col):
        return dataset_text_col.apply(lambda x: self.attack_note(x['text'], x[self.column]), axis = 1)
    
    def get_attacked_sample(self, source, N, delete = False):
        sample = source.obtain_samples(N).copy()
        sample['text'] = self.attack_sample(sample)
        return sample
    
class ProbabilisticTranslateNoteAttack(PerturbationAttack):
    def __init__(self, source, prob = None, root = "../../"):
        self.source = source
        self.prob = prob
        self.cache_filename = root+"translated_notes_cache.csv"
        self.cache = pd.read_csv(self.cache_filename)
        with open(root+"/api_key_gemini.txt", 'r') as f:
            self.api_key = f.readline()
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        
    def attack_note(self, note):
        raise Exception("attack_note shouldn't be getting called in this script")
        
        start = "You are a helpful medical assistant whose job is to make clinical notes more readable. Your goal is to remove clinical jargon and template-written portions of the notes.Rewrite this note so that it is more understandable: "
        return self.model.generate_content(start + note).text
        
    def get_fully_attacked_sample(self, notes_with_note_id):
        notes_with_note_id = notes_with_note_id.merge(self.cache, on = "note_id", how = "left")
        print(notes_with_note_id['translated'].isna().sum())
        alt_text = notes_with_note_id["translated"].copy()  # Copy existing translated text
        mask = notes_with_note_id["translated"].isna()  # Identify NaN rows

        # Apply attack_note only to rows where 'translated' is NaN
        alt_text.loc[mask] = notes_with_note_id.loc[mask, "text"].apply(self.attack_note)

        self.cache = pd.concat([self.cache, pd.DataFrame({\
                'note_id': notes_with_note_id.loc[mask, 'note_id'],\
                'translated': alt_text.loc[mask]\
            })\
        ])
        
        self.cache.to_csv(self.cache_filename)
        return alt_text

    def get_attacked_sample(self, source, N, delete = False):
        '''
        Slightly different. Instead of each note being attacked with probability p, exactly pN/N notes are being attacked. Notes
        have the same probability of being attacked but the probabilities are not independent.
        '''
        prob = self.prob
        if prob is None:
            prob = np.random.uniform()
        notes = source.obtain_samples(N).sample(frac = 1)
        notes.iloc[:int(N*prob), notes.columns.get_loc('text')] = self.get_fully_attacked_sample(notes.iloc[:int(N*prob), :]).values
        return notes
    
        