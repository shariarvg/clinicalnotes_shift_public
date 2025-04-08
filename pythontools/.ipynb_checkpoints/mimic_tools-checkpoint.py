'''
This file defines the MIMICEndpoint class, which is the easiest way to obtain subsetes of MIMIC-IV Note
'''

import pandas as pd
import numpy as np
import re
import sys
import google.generativeai as genai
from scipy.stats import chisquare
import random
import warnings

class MIMICEndpoint():
    
    def __init__(self, root = "../../..", path = "../../../notes_preproc.csv"):
        '''
        Access the diagnoses lookup, diagnoses by HADM, notes by HADM, admissions, and patients file
        Construct binary death outcome and admityear in admissions
        Construct start year and end year based on the anchor year group in patients
        '''
        self.diagnoses = pd.read_csv(root + "/physionet.org/files/mimiciv/3.0/hosp/diagnoses_icd.csv.gz")
        self.cache_info = {}
        if path is None:
            #self.di_lookup = pd.read_csv(root + "/physionet.org/files/mimiciv/3.0/hosp/d_icd_diagnoses.csv.gz")
            self.notes = pd.read_csv(root + "/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz")
            self.admissions = pd.read_csv(root + "//physionet.org/files/mimiciv/3.0/hosp/admissions.csv.gz")
            self.admissions['death'] = self.admissions['deathtime'].notna().astype(int)
            self.admissions['admittime'] = pd.to_datetime(self.admissions['admittime'])
            self.admissions['dischtime'] = pd.to_datetime(self.admissions['dischtime'])
            self.admissions['length_of_stay'] = (self.admissions['dischtime'] - self.admissions['admittime']).dt.days
            self.admissions['admityear'] = pd.to_datetime(self.admissions['admittime']).dt.year
            self.admissions = self.admissions.sort_values(by=['subject_id', 'admittime'])
            self.admissions['next_admittime'] = self.admissions.groupby('subject_id')['admittime'].shift(-1)
            self.admissions['days_until_next'] = (self.admissions['next_admittime'] - self.admissions['admittime']).dt.days
            self.admissions['admission_in_30_days'] = ((self.admissions['days_until_next'] > 0) & (self.admissions['days_until_next'] <= 30)).astype(int)

            self.patients = pd.read_csv(root + "/physionet.org/files/mimiciv/3.0/hosp/patients.csv.gz")
            self.patients['start_year'] = self.patients['anchor_year_group'].str[0:4].astype(int)
            self.patients['end_year'] = self.patients['anchor_year_group'].str[-4:].astype(int)
            
            
            self.patients['dod'] = pd.to_datetime(self.patients['dod'])
            self.admissions = self.admissions.merge(self.patients[['subject_id','dod']], on = 'subject_id', how = 'left')
            self.admissions['death_in_30_days'] = ((self.admissions['dod'] - self.admissions['dischtime']).dt.days < 30).astype(int)
            self.admissions[['ER','AST']] = pd.get_dummies(self.admissions['admission_location'])[['EMERGENCY ROOM','AMBULATORY SURGERY TRANSFER']]
            
            self.notes['over_65'] = self.notes['start_age'] > 65
            self.notes = self.merge_notes(self.notes)
            for i in self.notes.index: ## attempting to save memory
                self.notes.at[i, 'text'] = self.bhc(self.notes.at[i, 'text'], preprocess=False)
            self.notes = self.notes[(self.notes['text']!="")]
            for code in ['Z515', 'Z66','N170']:
                self.notes[f'has_{code}'] = self.notes['hadm_id'].apply(lambda x: x in self.patients[(self.patients['icd_code']==code)]['hadm_id'])
            
            self.notes.to_csv(root+"/notes_preproc.csv")
        else:
            self.patients = None
            self.notes = pd.read_csv(path)
            self.admissions = pd.read_csv(root + "//physionet.org/files/mimiciv/3.0/hosp/admissions.csv.gz")
        with open(root + "/everything/api_key_gemini.txt", 'r') as f:
                api_key = f.readline()

        genai.configure(api_key=api_key)

        self.genai_model = genai.GenerativeModel("gemini-1.5-flash")
        
    def truncate_corpus(self, max_words=50, notes = None, rand = False):
        if notes is None:
            notes = self.notes
        notes = notes.copy()
        if rand:
            notes['text'] = notes['text'].apply(lambda doc: ' '.join(random.sample(doc.split(),min(max_words, len(doc.split())))))
            return notes
        notes['text'] = notes['text'].apply(lambda doc: ' '.join(doc.split()[:max_words]))
        return notes
    
    def get_hadm_ids_sole_diagnosis(self):
        single_occ_hadm_ids = self.diagnoses['hadm_id'].value_counts()
        single_occ_hadm_ids = single_occ_hadm_ids[single_occ_hadm_ids==1].index.tolist()
        return single_occ_hadm_ids
        
    def get_hadmd_ids_version(self, version):
        hadmd_ids = set(self.diagnoses[(self.diagnoses["icd_version"]==version)]["hadm_id"])
        return hadmd_ids
        
    def get_hadmd_ids_diagnosis(self, code, version):
        '''
        Obtain the set of hadm id's associated with a diagnosis
        '''
        codes = code if isinstance(code, list) else [code]
        
        hadmd_ids = set(self.diagnoses[(self.diagnoses["icd_code"].isin(codes)) & (self.diagnoses["icd_version"]==version)]["hadm_id"]).intersection(set(self.notes['hadm_id']))
        return hadmd_ids
    
    def get_diagnoses_hadmd_ids(self, hadm_ids, version = 10):
        ad = self.diagnoses[(self.diagnoses['hadm_id'].isin(hadm_ids))]
        ad = ad[(ad['icd_version']==version)]
        return (ad['icd_code'].value_counts()/len(hadm_ids)).to_dict()
    
    def zstatistic(self, p1, p2, n1, n2):
        p = 1/(n1+n2) * (p1 * n1 + p2 * n2)
        return (p2 - p1)/np.sqrt(p*(1-p) * (1/n1 + 1/n2))
    
    def count_threshold(self, hadm_ids1, hadm_ids2, threshold=2, version = 10):
        diags1 = self.get_diagnoses_hadmd_ids(hadm_ids1, version)
        diags2 = self.get_diagnoses_hadmd_ids(hadm_ids2, version)
        s = 0
        n1 = len(hadm_ids1)
        n2 = len(hadm_ids2)
        for (key, value) in diags1.items():
            if key not in diags2.keys():
                s += int(abs(self.zstatistic(value, 0, n1, n2)) > threshold)
            else:
                s += int(abs(self.zstatistic(value, diags2[key], n1, n2)) > threshold)
        return s/len(hadm_ids1)
    
    def get_notes_start_age_greaterthan(self, age, notes = None, total_size = None):
        if notes is None:
            notes = self.notes
            
        return self.sample_with_total_size(notes[(notes['start_age'] > age)],total_size)
    
    def get_notes(self, total_size = None, notes = None):
        if notes is None:
            notes = self.notes
        return self.sample_with_total_size(notes, total_size)
    
    def get_notes_note_ids(self, note_ids, total_size = None, notes = None):
        if notes is None:
            notes = self.notes
            
        return self.sample_with_total_size(notes[(notes['note_id'].isin(note_ids))], total_size)
    
    def sample_with_total_size(self, notes, total_size):
        if total_size is None:
            return notes
        return notes.sample(total_size)
    
    def get_notes_key_value(self, k, v, greaterthan = False, lessthan = False, notes = None, total_size = None):
        if notes is None:
            notes = self.notes
            
        if greaterthan:
            return self.sample_with_total_size(notes[(notes[k] > v)],total_size)
        
        if lessthan:
            return self.sample_with_total_size(notes[(notes[k] < v)],total_size)
        
        return self.sample_with_total_size(notes[(notes[k] == v)],total_size)
    
    def get_notes_start_age_lessthanorequalto(self, age, notes = None, total_size = None):
        if notes is None:
            notes = self.notes
            
        return self.sample_with_total_size(notes[(notes['start_age'] <= age)], total_size)
    
    def get_hadmd_ids_syear(self, syear):
        return set(self.notes[(self.notes['start_year'] == syear)]['hadm_id'])
    
    def get_hadmd_ids_diagnosis_syear(self, code, version, syear):
        return self.get_hadmd_ids_diagnosis(code, version).intersection(self.get_hadmd_ids_syear(syear))
    
    def remove_word(self, word, notes = None):
        def remove_items(test_list, item): 
            res = [i for i in test_list if i != item] 
            return res 
        
        if notes is None:
            notes = self.notes
            
        notes['text'] = notes['text'].apply(lambda x: " ".join(remove_items(x.split(), word)))
        return notes
    
    def replace_word(self, word_to_replace, replacement_word, notes = None):
        def replace_items(test_list, w1, w2):
            def replace(string, item1, item2):
                if word == item1:
                    return item2
                return word
            return [replace(s, w1, w2) for s in test_list]
        
        if notes is None:
            notes = self.notes
            
        notes['text'] = notes['text'].apply(lambda x: " ".join(replace_items(x.split(), word_to_replace, replacement_word)))
        
        return notes
    

    def get_hadmd_id_sets(self, code1, code2, version1, version2):
        '''
        Obtain the set of hadm id's associated with diagnoses, not including intersections
        '''
        h1 = set(self.get_hadmd_ids_diagnosis(code1, version1))
        h2 = set(self.get_hadmd_ids_diagnosis(code2, version2))
        return h1-h2, h2-h1
    
    def get_hadmd_id_K_diagnosis(self, K):
        '''
        get all hadm id's that have K diagnoses. eg if K=3, get all hadm id's that have 3 unique diagnoses for it.
        '''
        hadm_counts = self.diagnoses['hadm_id'].value_counts()
        unique_hadm_ids = set(hadm_counts[hadm_counts == K].index)
        return unique_hadm_ids
    
    def generate_newnote_chatgpt(self, instruction1, instruction2, note, model_name = 'gpt-4'):
        '''
        Use the two instructions (one from system, one from user) to regenerate a note
        '''
        chat_completion = self.client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": instruction1
            },
            {
                "role": "user",
                "content": instruction2 + note
            }
        ],
        model=model_name,
    )
        
        return chat_completion.choices[0].message.content

    def generate_newtextcolumn_chatgpt(self, instruction1, instruction2, model_name = 'gpt-4', notes = None):
        '''
        Use the two instructions to regenerate a series of notes
        '''
        if notes is None:
            notes = self.notes
        return notes['text'].apply(lambda x: self.generate_newnote_chatgpt(instruction1, instruction2, x, model_name))
    
    def generate_readabletextcolumn_chatgpt(self, model_name = 'gpt-4', inplace = False, notes = None):
        '''
        Regenerate the text column, specifically to make it more readable for a patient.
        '''
        if notes is None:
            notes = self.notes
        
        instruction1 = "You are a helpful medical assistant whose job is to make clinical notes more readable."
        instruction2 = "Rewrite this note so that it is more understandable to a patient: "
        if inplace:
            self.notes['readable_text'] = notes['text'].apply(lambda x: self.generate_newnote_chatgpt(instruction1, instruction2, x, model_name))
            return
        
        return notes['text'].apply(lambda x: self.generate_newnote_chatgpt(instruction1, instruction2, x, model_name))
    
    
    def get_notes_K_diagnosis(self, K, subject_id_info = False, preprocess = False, total_size = None):
        unique_hadm_ids = self.get_hadmd_id_K_diagnosis(K)
        notes = self.get_notes_hadmd_ids(unique_hadm_ids)
        if subject_id_info:
            notes = self.add_subject_id(notes)
            notes = self.add_notes_after(notes)
        if total_size is not None:
            return notes.sample(total_size)
        return notes
    
    def get_diagnosis_counts(self, hadm_ids, version = 10):
        '''
        Get all diagnoses with frequencies from the list of hadm id's
        '''
        diagnoses = self.diagnoses[((self.diagnoses['hadm_id'].isin(hadm_ids)) & (self.diagnoses['icd_version']==10))]
        return diagnoses['icd_code'].value_counts()
    
    def get_codiagnoses_diagnoses(self, code, version = 10, year = None):
        '''
        Find the frequency of codiagnoses (this occurs when someone shares disease A and disease B) for a given diagnosis
        '''
        if year is None:
            hadm_ids = self.get_hadmd_ids_diagnosis(code, version)
        else:
            hadm_ids = self.get_hadmd_ids_diagnosis_syear(code, version, year)
        return self.get_diagnosis_counts(hadm_ids, version)
    
    def get_comorbidity_divergence(self, code, version, year1, year2, k = 5):
        '''
        Get top 5 codiagnoses from code in year1, with corresponding frequencies
        Find the corresponding frequencies of those codes in year2
        Take euclidean distance between them
        '''
        co1 = self.get_codiagnoses_diagnoses(code, version, year1)
        co1 = co1/co1.sum()
        
        co1_topk_index = co1.index[1:k+1]
        co1_topk_values = co1.values[1:k+1]
        
        co2 = self.get_codiagnoses_diagnoses(code, version, year2)
        co2 = co2/co2.sum()
        
        co2_co1_topk_values = co2.loc[co1_topk_index].values
        
        return np.linalg.norm((co1_topk_values - co2_co1_topk_values))
        
    def normalize_vc(self, vc1, vc2):
        p1 = vc1/vc1.sum()
        p2 = vc2/vc2.sum()
        return p1, p2
    
    def chisq(self, vc1, vc2):
        vc1, vc2 = vc1.align(vc2, fill_value = 0)
        p1, p2 = self.normalize_vc(vc1, vc2)
        return chisquare(f_obs = vc1, f_exp = p2*vc1.sum()).statistic
    
    def chisq_codiagnoses(self, code1, code2, version, k = 5):
        vc1 = self.get_codiagnoses_diagnoses(code1, version)
        vc2 = self.get_codiagnoses_diagnoses(code2, version)
        vc1 = vc1.drop([code1, code2], errors = 'ignore')
        vc2 = vc2.drop([code1, code2], errors = 'ignore')
        
        vc1_topk_index = vc1.index[1:k+1]
        
        vc2_vc1_topk = vc2.reindex(vc1_topk_index, fill_value=0)
        
        return self.chisq(vc1.loc[vc1_topk_index], vc2_vc1_topk)
    
    def chisq_codiagnoses_same_code(self, code, version, year1, year2, k= 5):
        vc1 = self.get_codiagnoses_diagnoses(code, version, year1)
        vc2 = self.get_codiagnoses_diagnoses(code, version, year2)
        vc1 = vc1.drop([code])
        vc2 = vc2.drop([code])
        
        vc1_topk_index = vc1.index[1:k+1]
        
        vc2_vc1_topk = vc2.reindex(vc1_topk_index, fill_value=0)
        print(self.chisq(vc1.loc[vc1_topk_index], vc2_vc1_topk))
        return self.chisq(vc1.loc[vc1_topk_index], vc2_vc1_topk)
    
    def get_notes_start_year(self, syear, total_size = None, preprocess = False, notes = None):
        '''
        Return K notes (or all notes) with start year syear, after merging with patients
        '''
        if notes is None:
            notes = self.notes
            
        if not (isinstance(syear, set) or isinstance(syear, list)):
            syear = set([syear])
            
        notes_syear = notes[(notes['start_year'].isin(syear))]
        
        if total_size is not None:
            return notes_syear.sample(min(total_size, notes_syear.shape[0]))
        return notes_syear
    
    def sample_total_size(self, notes, total_size = None):
        if total_size is None:
            return notes
        if total_size > notes.shape[0]:
            warnings.warn(f"total size is {total_size} but notes has {notes.shape[1]} rows", category=UserWarning, stacklevel=1)
            return notes
            
        return notes.sample(total_size)
    

    
    def merge_notes(self, n):
        '''
        Merge notes with patients, then create start year for a hadm id based on the anchor years and year
        '''
        n = n.merge(self.admissions[['hadm_id', 'admityear','death','admission_in_30_days','length_of_stay','death_in_30_days','ER','AST']], on = "hadm_id", how = "left")
        n = n.merge(self.patients[['subject_id', 'start_year', 'end_year', 'anchor_year','anchor_age','gender']], on = 'subject_id', how = 'left')
        n['start_year'] =n['start_year'] + n['admityear'] - n['anchor_year']
        n['end_year'] = n['end_year'] + n['admityear'] - n['anchor_year']
        n['start_age'] = n['anchor_age'] + n['admityear'] - n['anchor_year']
        return n
    
    def add_subject_id(self, notes):
        notes['subject_id_count'] = notes.groupby('subject_id')['subject_id'].transform('count')
        return notes
    def add_notes_after(self, notes):
        '''
        This column represents how many notes after this row have the same subject_id. 
        '''
        notes['notes_after'] = (
            self.notes[::-1]  # Reverse the DataFrame
            .groupby('subject_id').cumcount(ascending=True)  # Count occurrences per subject_id
            .iloc[::-1]  # Reverse the result back to match the original order
            .reset_index(drop=True)  # Reset the index to align with the original DataFrame
        )
        return notes
    
    def get_top_K_codes_icd10(self, K):
        d_icd10 = self.diagnoses[(self.diagnoses['icd_version']==10)]
        top_icd10_codes = d_icd10['icd_code'].value_counts()
        return list(top_icd10_codes.index[:K])

    def get_notes_diagnosis(self, code, version=10, total_size = None, notes = None):
        '''
        Get all notes for a diagnosis, obtain the BHC, remove empty notes, and then use merge_notes to merge with admissions and patients
        '''
        if notes is None:
            notes = self.notes
        if f"hadm_ids_diagnosis_{code}_{version}" not in self.cache_info.keys():
            self.cache_info[f"hadm_ids_diagnosis_{code}_{version}"] = self.get_hadmd_ids_diagnosis(code, version)
        notes = self.get_notes_hadmd_ids(self.cache_info[f"hadm_ids_diagnosis_{code}_{version}"], notes = notes)
        if total_size is not None:
            if notes.shape[0] < total_size:
                return None
            return notes.sample(total_size)
        return notes
    
    def get_notes_key_equals_value(self, key, value, total_size, bhc = True, preprocess = False, notes = None):
        '''
        Get total_size rows of notes (or self.notes) for which the column key has value value
        '''
        if notes is not None:
            return notes[(notes[key]==value)].sample(total_size)
        return self.notes[(self.notes[key]==value)].sample(total_size)
    
    def get_notes_key_equals_values(self, key, values, weights, total_size, bhc = True, preprocess = False, notes = None):
        '''
        Get notes from a mixture, where a row's key is of value values[i] with weight weights[i]
        '''
        ret_df = pd.DataFrame()
        for i, value in enumerate(values):
            ret_df = pd.concat([ret_df, self.get_notes_key_equals_value(key, value, int(total_size * weights[i]), bhc, preprocess, notes)])
        return ret_df
    
    def get_notes_key_greaterthan_value(self, key, value, total_size, strict = True, bhc = True, preprocess = False, notes = None):
        '''
        Get total_size rows of notes (or self.notes) for which the column key has value > (or >= if not strict) value
        '''
        if notes is not None:
            if strict:
                return notes[(notes[key]>value)].sample(total_size)
            return notes[(notes[key]>=value)].sample(total_size)
        if strict:
            return self.notes[(self.notes[key]>value)].sample(total_size)
        return self.notes[(self.notes[key]>=value)].sample(total_size)
    
    def get_notes_key_lessthan_value(self, key, value, total_size, strict = True, bhc = True, preprocess = False, notes = None):
        if notes is not None:
            if strict:
                return notes[(notes[key]<value)].sample(total_size)
            return notes[(notes[key]<=value)].sample(total_size)
        if strict:
            return self.notes[(self.notes[key]<value)].sample(total_size)
        return self.notes[(self.notes[key]<=value)].sample(total_size)
    
    def get_notes_key_split_value(self, key, value, weight_greater, total_size, strict_greater, bhc = True, preprocess = False, notes = None):
        '''
        Get mixture, where weight_greater percent of notes have key > value and 1 - weight_greater percent of notes have key < value. strict_greater decides exacting case
        '''
        ret_df = self.get_notes_key_greaterthan_value(key, value, int(weight_greater*total_size), strict_greater, bhc, preprocess, notes)
        ret_df = pd.concat([ret_df, self.get_notes_key_lessthan_value(key, value, total_size - ret_df.shape[0], not strict_greater, bhc, preprocess, notes)])
        return ret_df
    
    def count_notes_with_jargon(self, jargon_keywords, notes = None):
        if notes is None:
            notes= self.notes
        def note_has_jargon(note):
            return not set([n.lower() for n in note.split()]).isdisjoint(jargon_keywords)
        return notes['text'].apply(lambda x: note_has_jargon(x)).sum()
    
    def get_notes_without_jargon(self, jargon_keywords, notes = None):
        if notes is None:
            notes = self.notes
        def note_has_jargon(note):
                return not set([n.lower() for n in note.split()]).isdisjoint(jargon_keywords)   
        notes['has_jargon'] = notes['text'].apply(lambda x: note_has_jargon(x))
        return notes[~(notes['has_jargon'])]
    
    def avg_perc_jargon(self, jargon_keywords, notes = None):
        def perc_jargon(note):
            if len(note.split()) == 0:
                return np.nan
            return len([n for n in note.split() if n in jargon_keywords])/len(note.split())
        
        if notes is None:
            return self.notes['text'].apply(lambda x: perc_jargon(x)).mean()
        return notes['text'].apply(lambda x: perc_jargon(x)).mean()
                              
    
    def count_notes_with_keyword(self, keyword, split = True, notes = None):
        '''
        How many notes have this keyword? Note split is true here by default, but false by default
        in the accessor for the actual notes
        '''
        if notes is None:
            if split is True:
                return self.notes[self.notes['text'].apply(lambda x: keyword.lower() in [a.lower() for a in x.split()])].shape[0]
            return self.notes[(self.notes.text.str.contains(keyword))].shape[0]
        if split is True:
            return notes[notes['text'].apply(lambda x: keyword.lower() in [a.lower() for a in x.split()])].shape[0]
        return notes[(notes.text.str.contains(keyword))].shape[0]
    
    def get_notes_with_keyword(self, keyword, total_size=None, split = False, notes = None):
        '''
        Source for getting notes (particularly bhc's) with a desired keyword, such as ACTIVE ISSUES
        '''
        if notes is None:
            if split is True:
                return self.sample_total_size(self.notes[self.notes.apply(lambda x: keyword in x.text.split())],total_size)
            return self.sample_total_size(self.notes[(self.notes.text.str.contains(keyword))],total_size)
        if split is True:
            return self.sample_total_size(self.notes[self.notes.apply(lambda x: keyword in x.text.split())],total_size)
        return self.sample_total_size(self.notes[(self.notes.text.str.contains(keyword))],total_size)
    
    def balanced(self, notes, task, total, random = False):
        if random:
            notes = notes.sample(frac = 1)
        pos = notes[(notes[task]==1)].iloc[:int(total/2)]
        neg = notes[(notes[task]==0)].iloc[:int(total/2)]
        return pd.concat([pos, neg])
    
    def largest_balanced(self, notes, task, random = True):
        notes[task] = notes[task].astype(int)
        shapes = [notes[task].sum(), (1 - notes[task]).sum()]
        minority_class = np.argmin(shapes)
        
        if random:
            notes = notes.sample(frac = 1)
            
        notes = pd.concat([notes[(notes[task]==minority_class)], notes[(notes[task]==1-minority_class)].sample(min(shapes))])
        
        return notes
    
    def count_keyword_appearances(self, keyword, notes = None):
        if notes is None:
            notes = self.notes
        occurrences = np.sum([len(re.findall(rf'\b{keyword}\b', doc, flags=re.IGNORECASE)) for doc in list(notes['text'])])
        return occurrences
        
    
    def total_length(self, notes = None):
        if notes is None:
            notes = self.notes
        return notes['text'].apply(lambda x: len(x.split(" "))).sum()

    
    def get_notes_without_keyword(self, keyword, total_size = None, notes = None):
        if notes is None:
            return self.notes[(~(self.notes.text.str.contains(keyword)))].sample(total_size)
        return notes[(~(notes.text.str.contains(keyword)))].sample(total_size)
    
    def get_notes_version(self, version, total_size):
        '''
        Accessing total_size rows of notes (or self.notes)
        '''
        hadmd_ids = self.get_hadmd_ids_version(version)
        if total_size is None:
            return self.get_notes_hadmd_ids(hadmd_ids)
        return self.get_notes_hadmd_ids(hadmd_ids).sample(total_size)
    
    def get_mixture(self, codes, versions, weights, total_size, notes = None):
        '''
        Get a mixture of dataframes from different sources. Each "source" is the set of notes from a unique code.
        '''
        ret_df = pd.DataFrame()
        for i, code in enumerate(codes):
            if notes is None or i!=0:
                if code == "ANY":
                    new_df = self.get_notes(int(weights[i]*total_size))
                else:
                    new_df = self.get_notes_diagnosis(code, versions[i])
                    if new_df.shape[0] < int(weights[i]*total_size):
                        return pd.DataFrame()
                    new_df = new_df.sample(int(weights[i]*total_size))
            else:
                new_df = notes.sample(int(weights[i]*total_size))
            ret_df = pd.concat([ret_df, new_df])
        return ret_df
    
    def get_mixture_num_diagnoses(self, Ks, weights, total_size):
        '''
        Get a mixture of dataframes from different sources. Each "source" is the set of notes that have K diagnoses. Each element of Ks is a unique number of K.
        
        example: Ks = [2,4], weights = [0.7, 0.3]. 70% of notes will have 2 diagnoses, 30% will have 4 diagnoses.
        '''
        ret_df = pd.DataFrame()
        for i, K in enumerate(Ks):
            new_df = self.get_notes_K_diagnosis(K)
            new_df = new_df.sample(int(weights[i]*total_size))
            ret_df = pd.concat([ret_df, new_df])
        return ret_df
    
    def get_pure_and_mixture(self, codes, versions, weights, total_size):
        notes = self.get_notes_diagnosis(codes[0], versions[0]).sample(frac = 1)
        if notes.shape[0] < 2*total_size:
            return pd.DataFrame(), pd.DataFrame()
        return notes.iloc[:total_size], self.get_mixture(codes, versions, weights, total_size, notes = notes.iloc[total_size:])
        
    def get_notes_hadmd_ids(self, hadmd_ids, notes = None):
        '''
        Get all notes associated with a set of hadm id's
        '''
        if notes is None:
            notes = self.notes
        return notes[(notes["hadm_id"].isin(hadmd_ids))]

    def preprocess_text(self, text):
        '''
        (mostly deprecated) method for lowercasing all words and removing punctuation and splitting into words
        '''
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text)
        return words

    def get_words_corpus(self, corpus, vocab):
        '''
        Get the occurrence frequencies of every word in a corpus
        '''
        word_counts = {v: 0.0 for v in vocab}
        total_length = 0.0
        for document in corpus:
            pt = preprocess_text(document)
            for word in pt:
                if word in vocab:
                    word_counts[word] += 1
            total_length += len(pt)
        return np.array(list(word_counts.values())) / total_length

    def bhc(self, note, preprocess = False):
        '''
        Extract the BHC from a note
        '''
        if "Brief Hospital Course:\n" in note and "Medications on Admission" in note: 
            index1 = note.index("Brief Hospital Course:\n") + len("Brief Hospital Course:\n")
            index2 = note.index("Medications on Admission")
            if preprocess:
                return self.preprocess_text(note[index1:index2])
            return note[index1:index2]
        return ""
    
    def get_translated_note(self, note):
        start = "You are a helpful medical assistant whose job is to make clinical notes more readable. Your goal is to remove clinical jargon and template-written portions of the notes.Rewrite this note so that it is more understandable: "
        return self.genai_model.generate_content(start + note).text

    def get_translated_notes(self, notes):
        return [self.get_translated_note(note) for note in notes]

    def df(self, dataset, word, split_on_space= True):
        '''
        Get the document frequency of a word in a dataset
        '''
        if split_on_space:
            return sum([1.0 for doc in dataset['text'] if word in doc.lower().split(" ")])/len(dataset)
        return sum([1.0 for doc in dataset['text'] if word in doc.lower()])/len(dataset)

    def df_vector(self, dataset, vocabulary):
        '''
        Get the document frequency of every word in the vocabulary, for a given dataset
        '''
        return [df(dataset, word) for word in vocabulary]

    def get_statistic(self, p_hat_1, p_hat_2, N_1, N_2):
        '''
        Test statistic for two probabilities and counts
        '''
        p = (p_hat_1 * N_1 + p_hat_2 * N_2)/(N_1 + N_2)
        return (p_hat_1 - p_hat_2)/(np.sqrt(p*(1-p)/N_1 + p*(1-p)/N_2))
    
    def get_notes_los_greater_than(self, l, total_size, or_equal = False, notes = None):
        if notes is None:
            notes = self.notes
        if or_equal:
            return notes[(notes['length_of_stay'] >= l)].sample(total_size)
        return notes[(notes['length_of_stay'] > l)].sample(total_size)
    
    def delete_notes(self, notes):
        if notes is not None:
            assert self.notes is not None, "MIMIC Endpoint has no remaining notes!"
            self.notes = self.notes[~self.notes['hadm_id'].isin(notes['hadm_id'])]
            #self.diagnoses = self.diagnoses[~self.diagnoses['hadm_id'].isin(notes['hadm_id'])]
    
    def get_notes_los_less_than(self, l, total_size, or_equal = False, notes = None):
        if notes is None:
            notes = self.notes
        if or_equal:
            return notes[(notes['length_of_stay'] <= l)].sample(total_size)
        return notes[(notes['length_of_stay'] < l)].sample(total_size)
    
    def get_notes_los_range(self, l, r, total_size, notes = None):
        if notes is None:
            notes = self.notes
                
        if r == "inf":
            return self.get_notes_los_greater_than(l, total_size, or_equal = True, notes = notes)
        
        return notes[((notes['length_of_stay']>= l) & (notes['length_of_stay'] < r))].sample(total_size)
    
    def get_notes_diagnosis_death(self, code, version, death, total_size = None, notes = None):
        if notes is None:
            notes = self.notes
        notes = self.get_notes_diagnosis(code, version, notes = notes)
        return self.sample_with_total_size(notes[(notes['death'] == death)], total_size)
    
    
    def get_notes_death(self, death, total_size = None, notes = None):
        if notes is None:
            notes = self.notes
        if total_size is None:
            return notes[(notes['death'] == death)]
        return notes[(notes['death'] == death)].sample(total_size)
    
    def get_notes_readmission(self, readmission, total_size, notes = None):
        if notes is None:
            notes = self.notes
        if total_size is None:
            return notes[(notes['admission_in_30_days']==readmission)]
        return notes[(notes['admission_in_30_days']==readmission)].sample(total_size)
    
    def generate_icd_embedding(self, df_with_hadm_ids, icd_codes):
        diag = self.diagnoses[(self.diagnoses['icd_code'].isin(icd_codes))]
        pivot_table = pd.crosstab(diag['hadm_id'], diag['icd_code']).reindex(columns=icd_codes, fill_value=0)
        df_with_hadm_ids = df_with_hadm_ids.merge(pivot_table, how='left', left_on='hadm_id', right_index=True).fillna(0)
        return df_with_hadm_ids

# Ensure columns are binary (1 or 0)
    
    def get_notes_readmitted(self, readmission, total_size, notes = None):
        if notes is None:
            notes = self.notes
        return notes[(notes['admission_in_30_days']==readmission)].sample(total_size)
        
            
        