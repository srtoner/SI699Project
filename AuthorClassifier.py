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

import os
import json
import pandas as pd
import numpy as np
import pickle as pkl
import seaborn as sns

import torch

import torch.nn as nn
import torch.nn.functional as F
# load config
with open('config.json', 'r') as f:
    config = json.load(f)
cwd = os.getcwd()
os.chdir(config['REPODIR'])
import Utils as U
# from Corpus import Corpus
os.chdir(cwd)

from collections import Counter, defaultdict
import itertools


from tqdm.auto import tqdm, trange
from collections import Counter
import random
from torch import optim

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

suffix = "full"
# +
with open('sentence_embed.pkl', 'rb') as f:
    embed = pkl.load(f)

embed_df = pd.DataFrame(embed)
# -

embed_df = embed_df.rename(columns = {0: 'seqid', 1: 'passage_key', 2: 'sent_embeddings'})

data = U.load_file('data_vFFFF.pkl', 'pkl', config['DATADIR'])

data_df = pd.DataFrame(data)
data_df.head()

embed_df = embed_df.merge(data_df, how= 'left', left_on= 'passage_key', right_on = 'passage_key')

embed_df.head()

# +

n_classes = embed_df.author_id.nunique()

from sklearn.preprocessing import OneHotEncoder
# label_encoder=OneHotEncoder(sparse_output=False)
label_encoder=OneHotEncoder()

# -

y= label_encoder.fit_transform(embed_df['author_id'].to_numpy(dtype='int32').reshape(-1,1))
y = y.toarray()
X = embed_df['sent_embeddings']

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



device = 'cpu'
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if (device == "cuda:0" or device == 'mps') else {}
collate_func = lambda x: tuple(x_.to(device) for x_ in default_collate(x)) if device != "cpu" else None



class DocumentAttentionClassifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_size, num_heads, embeddings_fname, n_classes):
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
        self.linear = nn.Linear(num_heads * embedding_size, n_classes)

        self.attention = torch.rand(self.num_heads, self.embedding_size, requires_grad = True, device=device)
        
    def forward(self, w):
        w = w.squeeze()
        # w = torch.t(self.embeddings(word_ids).squeeze()) # Embedding_Dim 
        r = torch.matmul(self.attention, w)
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
val_list = datasets['test']

model = DocumentAttentionClassifier(1, 100, 4, 'trained_model_final', n_classes)
model = model.to(device)


# -

def run_eval(model, eval_data, kwargs):
    '''
    Scores the model on the evaluation data and returns the F1
    Eval Data must be in DataLoader-ready format
    '''

    eval_loader = DataLoader(eval_data, batch_size = 1, shuffle = False, collate_fn=collate_func, **kwargs)

    threshold = 0.2
    probs  = np.zeros(len(eval_loader))
    labels = []
    
    with torch.no_grad():
        for idx, x in enumerate(eval_loader):
            word_ids, label = x
            labels.append(label.cpu().numpy())
            output, weights = model(word_ids)
            probs[idx] = output.cpu().numpy()
    
    
    y_pred = np.array([1 if p >= threshold else 0 for p in probs], dtype = int)
    labels = np.array(labels)
    
    return labels, y_pred, f1_score(labels, y_pred, average='micro')

# +

loss_period = 500
# model = model.to(device)
writer = SummaryWriter()
loss_function = nn.CrossEntropyLoss()

# VVV GOLD STANDARD VVV
optimizer = optim.AdamW(model.parameters(), lr = 5e-3, weight_decay = 0.01)
# ^^^ GOLD STANDARD ^^^

# optimizer = optim.AdamW(model.parameters(), lr = 5e-3, weight_decay = 0.001)

# optimizer = optim.AdamW(model.parameters())
# optimizer = optim.RMSprop(model.parameters(), 5e-3)
# optimizer = optim.SGD(model.parameters(), lr = 5e-4)

train_loader = DataLoader(train_list, batch_size=16, shuffle=True, collate_fn=collate_func, **kwargs)
n_epochs = 10
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
        loss = loss_function(output, labels.float())
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        
        # TODO: Based on the details in the Homework PDF, periodically
        # report the running-sum of the loss to tensorboard. Be sure
        # to reset the running sum after reporting it.

        if not step % loss_period and step:
            writer.add_scalar("Loss", loss_sum, loss_idx)
            # if not step % (loss_period * 10) and step:
            #     model.eval()
            #     _y, _y2, f1 = run_eval(model, dev_list, kwargs)
            #     writer.add_scalar("F1", f1, loss_idx)
            #     model.train()
            loss_record.append(loss_sum)
            loss_sum = 0
            loss_idx += 1
            

        # TODO: it can be helpful to add some early stopping here after
        # a fixed number of steps (e.g., if step > max_steps)
        

# once you finish training, it's good practice to switch to eval.
model.eval()

y_true, y_pred, f1 = run_eval(model, val_list, kwargs)
print("Eval F1 Score of : "+ str(f1))
# -


y_true, y_pred, f1 = run_eval(model, test_list, kwargs)
print("Test F1 Score of : "+ str(f1))

torch.save(optimizer.state_dict(), 'trained_opt_' + suffix)
torch.save(model.state_dict(), 'trained_model_' + suffix)