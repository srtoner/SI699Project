import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from tqdm.auto import tqdm, trange
from collections import Counter
import random
from torch import optim
import json

import string
import re
from scipy import sparse
import Utils as U

from torch.utils.tensorboard import SummaryWriter

# Helpful for computing cosine similarity--Note that this is NOT a similarity!
from scipy.spatial.distance import cosine

# Handy command-line argument parsing
import argparse

# Sort of smart tokenization
from nltk.tokenize import RegexpTokenizer

# We'll use this to save our models
from gensim.models import KeyedVectors
import os
import pickle as pkl

URL_regex = 'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)'
twitter_username_re = '(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)'


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

# + vscode={"languageId": "python"}
class Corpus:

    def __init__(self):

        self.tokenizer = RegexpTokenizer(r'\w+')

        self.prob_table = {}

        self.word_to_index = {} # word to unique-id
        self.index_to_word = {} # unique-id to word

        # How many times each word occurs in our data after filtering
        self.word_counts = Counter()
        self.negative_sampling_table = []
        self.term_freq = None
        
        # The dataset we'll use for training, as a sequence of unique word ids
        self.full_token_sequence_as_ids = None
        self.unknown_token = '<UNK>'

    def tokenize(self, text):
        '''
        Tokenize the document and returns a list of the tokens
        '''
        return self.tokenizer.tokenize(text)

        # return text.apply(self.tokenizer.tokenize)        

    def load_data(self, file_name, min_token_freq, text = None):
        '''
        Reads the data from the specified file as long sequence of text
        (ignoring line breaks) and populates the data structures of this
        word2vec object.
        '''
        print('Reading data and tokenizing')

        if text is None:
            with open(file_name, 'r') as file:
                text = file.read() # ignore line breaks

        print('Counting token frequencies')
        tokens = self.tokenize(text)
        # tokens = ' '.join([' '.join(t).lower() for t in self.tokenize(text)])
        # _V = Counter(tokens.split())
        _V = Counter(tokens)
        
        print("Performing minimum thresholding")

        unk_count = 0
        for key, val in _V.items():
            if val < min_token_freq:
                unk_count += val
            else:
                self.word_counts[key] = val
        
        self.word_counts[self.unknown_token] = unk_count
        self.word_to_index = {word:idx for idx, word in enumerate(self.word_counts.keys())}
        self.index_to_word = {idx:word for word, idx in self.word_to_index.items()}
        self.index_to_word[-1] = self.unknown_token # to avoid key error

        _N = sum([val for val in self.word_counts.values()])
        for word, idx in self.word_to_index.items():
            self.prob_table[idx] = (
                np.sqrt(self.word_counts[word]/(_N *0.001)) + 1) \
              * (_N *0.001) / max(self.word_counts[word], 1)

        self.full_token_sequence_as_ids = [self.word_to_index[w] \
                                     if w in self.word_counts 
                                     else self.word_to_index[self.unknown_token] 
                                     for w in tokens]

        subsampled_seq = []

        for t in self.full_token_sequence_as_ids:
            if self.prob_table[t] > np.random.uniform():
                subsampled_seq.append(t)

        self.full_token_sequence_as_ids = subsampled_seq

        # Helpful print statement to verify what you've loaded
        print('Loaded all data from %s; saw %d tokens (%d unique)' \
              % (file_name, len(self.full_token_sequence_as_ids),
                 len(self.word_to_index)))
        
    def generate_negative_sampling_table(self, exp_power=0.75, table_size=1e6):
        '''
        Generates a big list data structure that we can quickly randomly index into
        in order to select a negative training example (i.e., a word that was
        *not* present in the context). 
        '''       

        print("Generating sampling table")
        
        _N = sum([val for val in self.word_counts.values()])
        freq = np.array([val / _N for val in self.word_counts.values()])
        power_arr = np.ones(freq.shape)
        power_arr.fill(exp_power)
        
        probs = np.power(freq, power_arr) / \
                np.sum(np.power(freq, power_arr))

        self.negative_sampling_table = np.random.choice(
            a=np.array([val for val in self.word_to_index.values()],dtype = int),
            size = int(table_size),
            p=probs
        )

    def generate_negative_samples(self, cur_context_word_id, num_samples):
        '''
        Randomly samples the specified number of negative samples from the lookup
        table and returns this list of IDs as a numpy array. As a performance
        improvement, avoid sampling a negative example that has the same ID as
        the current positive context word.
        '''
        results = []
        i = 0
        while i < num_samples:
            neg_sample = np.random.choice(self.negative_sampling_table)
            if neg_sample != cur_context_word_id:
                i += 1
                results.append(neg_sample)
        return results
    
    def consolidate_dict(self, corpus, stopword_file = None, min_freq = 2):
        # Corpus in this instance should be a pandas df or series
        stopwords = []
        if stopword_file:
            with open(stopword_file, 'r') as f:
                stopwords = f.readlines()

        self.term_freq = list(map(lambda x: {t: x.count(t) for t in x}, 
                        [[word.lower() for word in doc] for doc in self.tokenize(corpus)]))
        self.master_dict = {}
        
        for t in self.term_freq:
            for k in t.keys():
                if k not in stopwords:
                    if not k in self.master_dict:
                        self.master_dict[k] = []
                    self.master_dict[k].append(t[k])
        return self.word_counts, self.term_freq
    
    def DT_sparse_matrix(self, term_freq, numpy = False):
        # D x V matrix, so Document - Term Matrix
        V = sorted([key for key in self.word_counts.keys()])
        DT_matrix = np.zeros((len(term_freq), len(V))) 
        for d_idx, d in enumerate(term_freq):
            for w, c in d.items():
                if w in V:
                    # TODO: Change to use word2idx
                    DT_matrix[(d_idx, self.word_to_index[w])] = c
        if numpy:
            s_mat = sparse.csr_matrix(np.hstack((np.ones((len(term_freq),1)), DT_matrix)))
        else:
            s_mat = sparse.csr_matrix(DT_matrix)
            coord_list = s_mat.tocoo()
            idx = torch.LongTensor(np.vstack((coord_list.row, coord_list.col)))
            val = torch.FloatTensor(coord_list.data)
            s_mat = torch.sparse.LongTensor(idx, val, torch.Size(coord_list.shape))

        self.dt_s_mat = s_mat
        return s_mat
    
    def coincidence_matrix(self, term_freq, numpy = False):
        # How often does word V occur in the same context as V'
        V = sorted([key for key in self.word_counts.keys()])
        CO_matrix = np.zeros((len(V), len(V)))
        for d_idx, d in enumerate(term_freq):
            tokens = [k for k in d.keys()]
            for i in tokens:
                for j in tokens:
                    if i != j and i in self.word_to_index and j in self.word_to_index:
                        # Double counts but whatever
                        CO_matrix[(self.word_to_index[i], self.word_to_index[j])] += 1

        if numpy:
            s_mat = sparse.csr_matrix(np.hstack((np.ones((len(V),1)), CO_matrix)))
        else:
            s_mat = sparse.csr_matrix(CO_matrix)

        self.co_s_mat = s_mat
        return s_mat
    
    def process_data(self, input):

        pass
    
if __name__ == "__main__":

    with open('config.json', 'r') as file:
        config = json.load(file)

        
    corpus = Corpus()

    min_token_freq = 2

    filtered_df = U.load_file('data_w_subj.csv', 'csv', config['DATADIR'])
    filtered_df["text"] = filtered_df["text"].str.replace('[{}]'.format(string.punctuation), ' ')

    corpus.load_data("", min_token_freq, filtered_df['text'])
    corpus.generate_negative_sampling_table()
    # corpus.load_data("", min_token_freq, filtered_df['text'].to_string())
    corpus.consolidate_dict(filtered_df['text'], min_freq = min_token_freq)
    corpus.DT_sparse_matrix(corpus.term_freq)
    corpus.coincidence_matrix(corpus.term_freq)
    print("End of Test")

    with open('corpus.pkl', 'wb') as f:
        pkl.dump(corpus, f)