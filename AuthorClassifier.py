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
import torch

import torch.nn as nn
import torch.nn.functional as F
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


from tqdm.auto import tqdm, trange
from collections import Counter
import random
from torch import optim
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="dark")
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import nltk
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

torch.set_default_dtype(torch.float32)


# +
with open('embedding_data.pkl', 'rb') as f:
    embed = pkl.load(f)

embed_df = pd.DataFrame(embed)


n_classes = embed_df.author_id.nunique()

from sklearn.preprocessing import OneHotEncoder
label_encoder=OneHotEncoder(sparse_output=False)

# -

y= label_encoder.fit_transform(embed_df['author_id'].to_numpy(dtype='int32').reshape(-1,1))
# X = embed_df['sent_embeddings']
X = embed_df['vectors'] # Word Embeddings

# +
test_size = 0.2
val_size = 0.2
random_state =699

X_train, X_test, y_train, y_test = U.train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y)

# Split train set into train and validation sets
X_train, X_val, y_train, y_val = U.train_test_split(X_train, y_train, test_size=val_size/(1-test_size),
                                                    random_state=random_state,
                                                    stratify=y_train)
# -

type(embed_df.sent_embeddings.iloc[0])

device = 'cpu'
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if (device == "cuda:0" or device == 'mps') else {}
collate_func = lambda x: tuple(x_.to(device) for x_ in default_collate(x)) #if device != "cpu" else default_collate

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


class DocumentAttentionClassifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_size, num_heads, hidden_dim, embeddings_fname, n_classes):
        '''
        Creates the new classifier model. embeddings_fname is a string containing the
        filename with the saved pytorch parameters (the state dict) for the Embedding
        object that should be used to initialize this class's word Embedding parameters
        '''
        super(DocumentAttentionClassifier, self).__init__()
        
        # Save the input arguments to the state
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.embeddings_fname = vocab_size        
        
        # Create the Embedding object that will hold our word embeddings that we
        # learned in word2vec. This embedding object should have the same size
        # as what we learned before. However, we don't to start from scratch! 
        # Once created, load the saved (word2vec-based) parameters into the object
        # using load_state_dict.

        # trained_weights = torch.load(embeddings_fname)['target_embeddings.weight']

        # self.embeddings = nn.Embedding.from_pretrained(trained_weights, freeze = False)
        # self.embeddings = nn.Embedding()
        self.lstm = nn.LSTM(embedding_size, hidden_dim, bidirectional=True)
        
        # self.attention = torch.rand(self.num_heads, self.embedding_size, requires_grad = True, device=device)
        self.attention = torch.rand(self.num_heads, hidden_dim * 2, requires_grad = True, device=device)
        self.linear = nn.Linear(num_heads * embedding_size, n_classes)
    
    def forward(self, w):
        w = w.squeeze()

        lstm_out, _ = self.lstm(w.T)
        # w = torch.t(self.embeddings(word_ids).squeeze()) # Embedding_Dim 
        r = torch.matmul(self.attention, lstm_out.T)
        a = torch.softmax(r, 1)
        reweighted = a @ w.T
        output = self.linear(reweighted.view(-1))

        return torch.softmax(output, dim=0), a.T


# +
datasets = {}

datasets['train'] = list(zip(X_train, y_train))
datasets['val'] = list(zip(X_val, y_val))
datasets['test'] = list(zip(X_test, y_test))

train_list = datasets['train']
val_list = datasets['val']

model = DocumentAttentionClassifier(1, 50, 4, 32, 'trained_model_final', n_classes)
model = model.to(device)


# -

def run_eval(model, eval_data, n_classes, kwargs):
    '''
    Scores the model on the evaluation data and returns the F1
    Eval Data must be in DataLoader-ready format
    '''

    eval_loader = DataLoader(eval_data, batch_size = 1, shuffle = False, collate_fn=collate_func, **kwargs)

    threshold = 0.2
    probs = np.zeros((len(eval_loader), n_classes))
    labels = []
    
    with torch.no_grad():
        for idx, x in enumerate(eval_loader):
            word_ids, label = x
            labels.append(label.cpu().numpy())
            output, weights = model(word_ids)
            probs[idx] = output.cpu().numpy()
    
    
    y_pred = np.array([np.argmax(p) for p in probs], dtype = int)
    labels = np.array(labels)

    y_true = [np.argmax(l) for l in labels]
    
    
    return labels, y_pred, f1_score(y_true, y_pred, average='micro')

# +

loss_period = 5
# model = model.to(device)
writer = SummaryWriter()
loss_function = nn.CrossEntropyLoss()

# VVV GOLD STANDARD VVV
optimizer = optim.AdamW(model.parameters(), lr = 5e-3, weight_decay = 0.1)
# ^^^ GOLD STANDARD ^^^

# optimizer = optim.AdamW(model.parameters(), lr = 5e-3, weight_decay = 0.1)

# optimizer = optim.AdamW(model.parameters())
# optimizer = optim.RMSprop(model.parameters(), 5e-3)
# optimizer = optim.SGD(model.parameters(), lr = 5e-4)

train_loader = DataLoader(train_list, batch_size=1, shuffle=True, collate_fn=collate_func, **kwargs)
n_epochs = 3
# n_epochs = 1

# # + vscode={"languageId": "python"}
loss_idx = 0
loss_record = []
model.train()

# # + vscode={"languageId": "python"}r
for epoch in tqdm(range(n_epochs)):

    loss_sum = 0

    for step, data in tqdm(enumerate(train_loader)):

        word_ids, labels = data
        model.train()
        model.zero_grad()
        output, weights = model(word_ids)
        loss = loss_function(output, labels.squeeze().float())
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        
        # TODO: Based on the details in the Homework PDF, periodically
        # report the running-sum of the loss to tensorboard. Be sure
        # to reset the running sum after reporting it.

        if not step % loss_period and step:
            writer.add_scalar("Loss", loss_sum, loss_idx)
            if not step % (loss_period * 10) and step:
                model.eval()
                _y, _y2, f1 = run_eval(model, val_list, n_classes, kwargs)
                writer.add_scalar("F1", f1, loss_idx)
                model.train()
            loss_record.append(loss_sum)
            loss_sum = 0
            loss_idx += 1
            

        # TODO: it can be helpful to add some early stopping here after
        # a fixed number of steps (e.g., if step > max_steps)
        

# once you finish training, it's good practice to switch to eval.
model.eval()

torch.save(optimizer.state_dict(), 'trained_opt_')
torch.save(model.state_dict(), 'trained_model_')

y_true, y_pred, f1 = run_eval(model, val_list, n_classes, kwargs)
print("F1 Score of : "+ str(f1))
# -

