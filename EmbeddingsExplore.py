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
# -

with open ('corpus_text.txt', 'r') as f:
    corpus_text = f.read()

length = len(corpus_text)

trunc = length * .3 // 1
corpus_text = corpus_text[:trunc]

with open ('corpus_text_new.txt', 'w') as f:
    f.write(corpus_text)


# # +
# subsample = data_df.sample(10000, random_state=0)

# subject_b = [s for s in subsample['Sub_B']]
# other_subject = [s if s != 'Fiction' else subject_b[idx] for idx, s in enumerate(subsample['Sub_A']) ]

# subsample['sub'] = other_subject
# subsample = subsample[subsample['authoryearofbirth'] > 1750]
# # -

# # Fetch works for every book

# data = subsample.to_dict(orient='records')

# # +
# data_text = [dat['text'] for dat in data]
# tokenizer = nltk.RegexpTokenizer(r'\w+')

# tokenized_text = [tokenizer.tokenize(' '.join(text)) for text in data_text]
# joined_text = [' '.join(text) for text in data_text]

# # +
# # Worked fine with 'glove-wiki-gigaword-300'

# import gensim.downloader
# glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
# # -

# for idx, d in enumerate(data):
#     d['vectors'] = []
#     for token in tokenized_text[idx]:
#         try:
#             d['vectors'].append(glove_vectors[token])
#         except:
#             continue
#     d['mean_vector'] = np.array([v for v in d['vectors']]).mean(axis = 0)
#     if not d['mean_vector'].shape:
#         d['mean_vector'] = np.zeros((100,))
#     d['key'] = str(d['book_id']) + str(d['text_lines'])
#     #  d['subject'] = catalog[catalog['book_id'] == d['book_id']].iloc[0, 9]

# # mean_vectors = np.array([d['mean_vector'] for d in data if d['mean_vector'].shape and d['subject'] != 'Other'])
# mean_vectors = np.array([d['mean_vector'] for d in data if d['mean_vector'].shape])



# # ids = [idx for idx, d in enumerate(data) if d['mean_vector'].shape and d['subject'] != 'Other']
# ids = [idx for idx, d in enumerate(data) if d['mean_vector'].shape]

# titles = [data[idx]['title_x'] for idx in ids]
# subjects = [data[idx]['sub'] for idx in ids]
# # decade = [int(data[idx]['decade']) for idx in ids]
# yob = [int(data[idx]['authoryearofbirth']) for idx in ids]
# # gender = [1 if data[idx]['gender'] == "F" else 0 for idx in ids]
# # penname =[1 if data[idx]['penname'] == "Y" else 0 for idx in ids]
# # pn_gend = [str(penname[idx]) + str(gender[idx]) for idx in ids]

# data[9]['text']

# two_dim = mean_vectors


# # two_dim = PCA(random_state =0).fit_transform(mean_vectors)[:1000,]
# # two_dim = PCA(random_state =0).fit_transform(mean_vectors)


# xPCA = two_dim[:,0]; yPCA = two_dim[:,1]
# xPCA.shape

# f, ax = plt.subplots(figsize=(6, 6))
# sns.scatterplot(x=xPCA, y=yPCA, s =5, color=".15")
# sns.histplot(x=xPCA, y=yPCA, bins=50, pthresh=.1, cmap="mako")
# sns.kdeplot(x=xPCA, y=yPCA, levels=5, color="w", linewidths=1)

# # +
# import plotly.express as px

# def pca_scatterplot_3D(model, user_input=None, color_map=None, sentences = None):

#     # three_dim = user_input[:,:3]
#     three_dim = PCA(random_state=0).fit_transform(user_input)[:,:3]

#     df = pd.DataFrame(three_dim)
#     df['Label'] = color_map
#     df['sentence'] = sentences

#     plot_figure = px.scatter_3d(df, x = 0, y = 1, z = 2, color = 'Label', hover_name = 'sentence', size_max = 2)

#     plot_figure.show()


# # +

# # For use in the famous literature set
# # decade_gender = [int(str(decade[idx]) + str(gender[idx])) for idx in ids]

# # +
# from sklearn import mixture

# gmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type="full").fit(two_dim)
# clusters = gmm.predict(mean_vectors)
# # -

# # pca_scatterplot_3D(data, user_input = two_dim, color_map = [str(c) for c in clusters], sentences = titles)
# pca_scatterplot_3D(data, user_input = two_dim, color_map = yob, sentences = subjects)


# # +
# from sklearn.cluster import AgglomerativeClustering

# clustering = AgglomerativeClustering(n_clusters = 6).fit(mean_vectors)

# cluster_labels = [str(lab) for lab in clustering.labels_]
# # -



# # +
# extract_data = []

# for i, idx in enumerate(ids):
#     data[idx]['cluster'] = str(clusters[i])
#     extract_data.append(data[idx].copy())


# # -

# t = pd.DataFrame(extract_data).groupby(by = ['cluster', 'sub']).agg('count')

# ed = pd.DataFrame(extract_data)

# # +
# # ed[ed['cluster'] == 3].unique()
# # -

# t['key'].unstack(0).divide(sum(t['key']))
