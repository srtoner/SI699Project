import pickle as pkl
import pandas as pd
import os
import numpy as np
from collections import Counter
import json

with open('config.json', 'r') as f:
    config = json.load(f)
cwd = os.getcwd()
os.chdir(config['REPODIR'])
import Utils as U
from Corpus import Corpus
os.chdir(cwd)

if __name__ == "__main__":

    suffix = 'start'
    save_pickle = True

    if not os.path.exists('data/corpus' + suffix + '.pkl'):
        corpus = Corpus()
        corpus.load_data('corpus_text.txt', 5)
        corpus.generate_negative_sampling_table()
        with open('corpus' + suffix + '.pkl', 'wb') as f:
            pkl.dump(corpus, f)        
    else:
        with open('data/corpus' + suffix + '.pkl', 'rb') as f:
            corpus = pkl.load(f)

    window_size = 2
    num_negative_samples_per_target = 2
    max_context = 2 * window_size

    training_data = []
    input_seq = corpus.full_token_sequence_as_ids

    for i in range(len(input_seq)):

        target_word = [input_seq[i]]
        # Skip over training instances that are <UNK>
        if corpus.index_to_word[input_seq[i]] == corpus.unknown_token:
            continue

        tail, head = max(i - window_size, 0), min(len(input_seq) - i, window_size)
        context = input_seq[tail:i] + input_seq[(i + 1):(head + i + 1)]
        # Remove subsampled words
        context = [c for c in context if c != -1]
        n_context = len(context)
        if not n_context:
            context = target_word
            n_context = 1

        # Determine how many context words are missing
        deficit = max_context - n_context
        samples = [c for c in context]
        labels = [1.0] * n_context

        # make up the deficit of positive samples with negative samples from the context
        if deficit:
            samples += corpus.generate_negative_samples(np.random.choice(samples), deficit)
            labels += [0.0] * deficit

        for c in range(max_context):
            samples += corpus.generate_negative_samples(samples[c],
                                num_negative_samples_per_target)
            labels += [0.0] * num_negative_samples_per_target

        training_data.append(
                                (
                                    np.array(target_word),
                                    np.array(samples),
                                    np.array(labels, dtype=int)
                                )
        )

    if not os.path.exists('training_data' + suffix + '.pkl') or save_pickle:
        with open('training_data' + suffix + '.pkl', 'wb') as f:
            pkl.dump(training_data, f)

    if not os.path.exists('corpus' + suffix + '.pkl'):
        with open('corpus' + suffix + '.pkl', 'wb') as f:
            pkl.dump(corpus, f)