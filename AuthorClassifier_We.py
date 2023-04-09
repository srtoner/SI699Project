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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from gensim.test.utils import common_texts
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

torch.set_default_dtype(torch.float64)

suffix = "small"
# +
with open('embedding_data_final.pkl', 'rb') as f:
    embed = pkl.load(f)

embed_df = pd.DataFrame(embed)
# -

embed_df = embed_df.rename(columns = {0: 'seqid', 1: 'passage_key', 2: 'sent_embeddings'})


data = U.load_file(f'data_vF_{suffix}.pkl', 'pkl', config['DATADIR'])

# +
import gensim

import gensim.downloader as api
gensim_model = api.load("glove-wiki-gigaword-300") 
# -

embedding_weights = gensim_model.vectors
# +


embedding_dim = gensim_model.vector_size
num_embeddings = embedding_weights.shape[0]

embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
embedding_layer.weight.data.copy_(torch.from_numpy(embedding_weights))
# -

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

# +
data_df = pd.DataFrame(data)
data_df.head()

data_df['joined_text'] = data_df['text'].apply(' '.join)
# -

text = data_df['joined_text'].to_numpy()
text_clean = [tokenizer.tokenize(t.lower()) for t in text]

word2index = {token: token_index for token_index, token in enumerate(gensim_model.index_to_key)}
index2word = {val:key for key, val in word2index.items()}

tokenized_sequences = [[word2index[word] for word in text if word in word2index ] for text in text_clean]

lengths = [len(seq) for seq in tokenized_sequences]
max_len = max(lengths)
min_len = min(lengths)



# +
test = tokenized_sequences[0]
# torch.IntTensor(tokenized_sequences)

data_df['sequences'] = tokenized_sequences

# +
# embed_df = embed_df.dropna(subset = ['author_id', 'sent_embedding'])

# +

n_classes = data_df.author_id.nunique()

from sklearn.preprocessing import OneHotEncoder
# label_encoder=OneHotEncoder(sparse_output=False)
label_encoder=OneHotEncoder()

# -

y= label_encoder.fit_transform(data_df['author_id'].to_numpy(dtype='int32').reshape(-1,1))
y = y.toarray()
X = data_df['sequences'].apply(np.array)

X = [(toop[0], toop[1]) for toop in list(zip(X, lengths))]

X[0]


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


# +
MAX_LENGTH = 128

X_trunc = []

for x in X:
    deficit = MAX_LENGTH - x[1]
    if deficit > 0:
        X_trunc.append(x[0])
    else:
        X_trunc.append(x[0][:MAX_LENGTH])

# -

X_trunc.sort(key=lambda x: len(x),reverse = True)

len(X_trunc[-1])

# +
torch.tensor(X_trunc[0]).shape

tensor_list = [torch.tensor(x).unsqueeze(1) for x in X_trunc]
# -

tensor_list[3].shape

X_t = pad_sequence([torch.tensor(x) for x in X_trunc], batch_first=True)


[xt.shape for xt in X_t]

# +
test_size = 0.2
val_size = 0.2
random_state =699

X_train, X_test, y_train, y_test = U.train_test_split(X_t, y, test_size=test_size,
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
collate_func = lambda x: tuple(x_.to(device) for x_ in default_collate(x)) if device != "cpu" else default_collate(x)



class DocumentAttentionClassifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_size, num_heads, hidden_dim, trained_weights, n_classes, sequence_length):
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
        
        self.embeddings = nn.Embedding.from_pretrained(trained_weights, freeze = False)
        self.lstm = nn.LSTM(sequence_length, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)
    
    def forward(self, text):
       
        w = self.embeddings(text)

        lstm_output, _ = self.lstm(w.transpose(1,2)) 
        attention_scores = self.attention(lstm_output) 
        attention_weights = F.softmax(attention_scores, dim=1)  
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  
        output = self.fc(context_vector)  # shape: (batch_size, num_classes)

        return output, attention_weights

# +
datasets = {}

datasets['train'] = list(zip(X_train, y_train))
datasets['val'] = list(zip(X_val, y_val))
datasets['test'] = list(zip(X_test, y_test))

train_list = datasets['train']
val_list = datasets['val']
test_list = datasets['test']

model = DocumentAttentionClassifier(1, 300, 4, 32, torch.Tensor(embedding_weights), n_classes, MAX_LENGTH)


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
            output, weights = model(word_ids.long())
            probs[idx] = output.cpu().numpy()
    
    
    y_pred = np.array([np.argmax(p) for p in probs], dtype = int)
    labels = np.array(labels)

    y_true = [np.argmax(l) for l in labels]
    
    
    return labels, y_pred, f1_score(y_true, y_pred, average='weighted')

# +

loss_period = 5
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

train_loader = DataLoader(train_list, batch_size=256, shuffle=True, collate_fn=collate_func, **kwargs)
n_epochs = 10

# # + vscode={"languageId": "python"}
loss_idx = 0
loss_record = []
model.train()

# # + vscode={"languageId": "python"}r
for epoch in tqdm(range(n_epochs)):

    loss_sum = 0

    for step, data in tqdm(enumerate(train_loader)):

        word_ids, labels = data
        labels = labels.argmax(-1)
        model.train()
        model.zero_grad()
        output, weights = model(word_ids)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

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
        
# once you finish training, it's good practice to switch to eval.
model.eval()

print(n_classes)

y_true, y_pred, f1 = run_eval(model, val_list, n_classes, kwargs)
print("Eval F1 Score of : "+ str(f1))
# -


y_true, y_pred, f1 = run_eval(model, test_list, n_classes, kwargs)
print("Test F1 Score of : "+ str(f1))

torch.save(optimizer.state_dict(), 'trained_opt_author' + suffix)
torch.save(model.state_dict(), 'trained_model_author' + suffix)
