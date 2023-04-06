# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: si699proj
#     language: python
#     name: python3
# ---

# +
from gutenbergpy.gutenbergcache import GutenbergCache, GutenbergCacheTypes
import os
import json
import pandas as pd
import numpy as np
import pickle as pkl
import seaborn as sns
import gensim
from gensim.test.utils import common_texts

# load config
with open('config.json', 'r') as f:
    config = json.load(f)
cwd = os.getcwd()
os.chdir(config['REPODIR'])
import Utils as U
from Corpus import Corpus
os.chdir(cwd)

from collections import Counter, defaultdict
import itertools

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="dark")
import pickle as pkl
from sklearn.decomposition import PCA
import nltk

# +
# with open("data_vFinal.pkl", "rb") as infile:
        # data = pkl.load(infile)

data = U.load_file('data_vFF.pkl', 'pkl', config['DATADIR'])

# Just table of contents
# data.pop(3064)
# -

data_df = pd.DataFrame(data)
data_df.head()
baby = int(data_df.author_id.nunique() * .15 // 1)
baby_set = set(data_df.author_id.sample(baby))
baby_df = data_df[data_df.author_id.apply(lambda X: X in baby_set)]
baby_df.to_csv('baby_data.csv', index = False)

# +
# data_df = data_df.rename(columns={
#     'title_y' : 'title',
#     'authoryearofbirth_x':'authoryearofbirth',
#     'authoryearofdeath_x':'authoryearofdeath',
#     'downloads_x':'downloads',
#     'subjects_x':'subjects',
#     'Sub_A_x':'topic',
#     'Sub_A_y':'Sub_A',
#     'Sub_B_y':'Sub_B',
#     'Sub_C_y':'Sub_C'
# })
# -

baby = baby_df.to_dict('records')

# +
subsample = data_df.sample(9000, random_state=0)

subject_b = [s for s in subsample['Sub_B']]
other_subject = [s if s != 'Fiction' else subject_b[idx] for idx, s in enumerate(subsample['Sub_A']) ]

subsample['sub'] = other_subject
subsample = subsample[subsample['authoryearofbirth'] > 1750]
# -

subsample.groupby('sub').agg({'book_id':'count'}).sort_values(by = 'book_id', ascending=False).head(20).plot(kind='bar')

# Fetch works for every book



# +
data_text = [dat['text'] for dat in baby]
tokenizer = nltk.RegexpTokenizer(r'\w+')

tokenized_text = [tokenizer.tokenize(' '.join(text)) for text in data_text]
joined_text = [' '.join(text) for text in data_text]

# +
# Worked fine with 'glove-wiki-gigaword-300'

import gensim.downloader
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
# -

for idx, d in enumerate(baby):
    d['vectors'] = []
    for token in tokenized_text[idx]:
        try:
            d['vectors'].append(glove_vectors[token])
        except:
            continue
    d['mean_vector'] = np.array([v for v in d['vectors']]).mean(axis = 0)
    if not d['mean_vector'].shape:
        d['mean_vector'] = np.zeros((100,))
    d['key'] = str(d['book_id']) + str(d['text_lines'])
    #  d['subject'] = catalog[catalog['book_id'] == d['book_id']].iloc[0, 9]

with open('word_embeddings_data.pkl', 'wb') as f:
    pkl.dump(baby, f)

