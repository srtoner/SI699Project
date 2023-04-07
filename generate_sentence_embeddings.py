import os
import json
import pandas as pd
import pickle as pkl
import itertools 

with open('config.json', 'r') as f:
    config = json.load(f)
cwd = os.getcwd()
os.chdir(config['REPODIR'])
import Utils as U
from Corpus import Corpus
os.chdir(cwd)

import gensim.downloader as api
from multiprocessing import Pool, cpu_count

glove_model = api.load('glove-wiki-gigaword-300')

 # Define the function to generate sentence embeddings
def generate_embedding(sentence):
    
    words = " ".join(sentence).lower().split()
    word_embeddings = [glove_model[word] for word in words if word in glove_model]
    if not word_embeddings:
        return None
    sentence_embedding = sum(word_embeddings) / len(word_embeddings)
    return sentence_embedding



if __name__ == '__main__':
    # Load pre-trained GloVe embeddings
   
    data = U.load_file('data_vFFF.pkl', 'pkl', config['DATADIR'])
    # data = data[:16]
    text_combined = list(itertools.chain.from_iterable([dat['text'] for dat in data]))
    passage_keys = list(itertools.chain.from_iterable([[dat['passage_key']]*len(dat['text']) for dat in data]))
    seq_idx = [i for i in range(len(text_combined))]

   


    # # Tokenize and preprocess your corpus
    # corpus = text_combined
    # # Define the number of processes to use
    # num_processes = cpu_count()

    # # Split the corpus into chunks
    # chunk_size = len(corpus) // num_processes
    # chunks = [corpus[i:i+chunk_size] for i in range(0, len(corpus), chunk_size)]

    # # Generate sentence embeddings in parallel
    # with Pool(num_processes) as pool:
    #     sentence_embeddings = []
    #     for chunk_embeddings in pool.map(generate_embedding, chunks, chunk_size):
    #         sentence_embeddings.extend(chunk_embeddings)

    # print("pause:")

    sentence_embeddings = []

    for text in text_combined:
        sentence_embeddings.append(generate_embedding(text))

    zipped = list(zip(seq_idx, passage_keys, text_combined, sentence_embeddings))

    # Save sentence embeddings to file
    with open('sentence_embeddings.pkl', 'wb') as f:
        pkl.dump(zipped, f)