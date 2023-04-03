import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from gensim.models import KeyedVectors

import json
import os

with open('config.json', 'r') as f:
    config = json.load(f)
cwd = os.getcwd()
os.chdir(config['REPODIR'])
import Utils as U
from Corpus import Corpus
os.chdir(cwd)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from tqdm.auto import tqdm, trange
from collections import Counter
import random
from torch import optim



from torch.utils.tensorboard import SummaryWriter

class Word2Vec(nn.Module):
    
    def __init__(self, vocab_size, embedding_size):
        super(Word2Vec, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.target_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.init_emb(init_range=0.5/self.vocab_size)
        
    def init_emb(self, init_range):

        init.uniform_(self.target_embeddings.weight, -init_range, init_range)
        init.uniform_(self.context_embeddings.weight, -init_range, init_range)
        
    def forward(self, target_word_id, context_word_ids):
        ''' 
        Predicts whether each context word was actually in the context of the target word.
        The input is a tensor with a single target word's id and a tensor containing each
        of the context words' ids (this includes both positive and negative examples).
        '''

        # Embedded target word
        h = self.target_embeddings(target_word_id) # Shape: batch size, 1, embedding_dim

        # Embedded Context words
        u = self.context_embeddings(context_word_ids) # 
        u = u.transpose(1, 2)

        product = torch.bmm(h,u)
        sum = torch.sum(product, dim=1)
        sig = torch.sigmoid(sum)
        return sig
    
def save(model, corpus, filename):
    '''
    Saves the model to the specified filename as a gensim KeyedVectors in the
    text format so you can load it separately.
    '''

    # Creates an empty KeyedVectors with our embedding size
    kv = KeyedVectors(vector_size=model.embedding_size)        
    vectors = []
    words = []
    # Get the list of words/vectors in a consistent order
    for index in trange(model.target_embeddings.num_embeddings):
        word = corpus.index_to_word[index]
        vectors.append(model.target_embeddings(torch.LongTensor([index]).to(device)).cpu().detach().numpy()[0])
        words.append(word)

    # Fills the KV object with our data in the right order
    kv.add_vectors(words, vectors) 
    kv.save_word2vec_format(filename, binary=False)

if __name__ == '__main__':

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    suffix = 'whew'
    save_pickle = True
    torch.set_default_dtype(torch.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda:0" else {}

    collate_func = default_collate
    # collate_func = lambda x: tuple(x_.to(device) for x_ in default_collate(x)) if device != "cpu" else default_collate
    print("Running on: " + str(device))

    corpus = U.load_file('corpus.pkl','pkl', config['DATADIR'])
    training_data = U.load_file('data_v2.pkl','pkl', config['DATADIR'])
    # 

    
    loss_period = 100
    model = Word2Vec(len(corpus.word_to_index), 50)
    model = model.to(device)
    writer = SummaryWriter()
    loss_function = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr = 5e-3, weight_decay = 1e-3)
    train_data = DataLoader(training_data, batch_size=512, shuffle=True, 
                collate_fn=collate_func,
                **kwargs)

    n_epochs = 2
    loss_idx = 0
    loss_record = []
    model.train()

    for epoch in tqdm(range(n_epochs)):
        loss_sum = 0
    
        for step, data in tqdm(enumerate(train_data)):
            model.train()
            model.zero_grad()
            target_ids, context_ids, labels = data

            output = model(target_ids, context_ids)
            loss = loss_function(output, labels.float())
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            
            if not step % loss_period and step:
                writer.add_scalar("Loss", loss_sum, loss_idx)
                loss_record.append(loss_sum)

                loss_sum = 0
                loss_idx += 1

    model.eval()

    
