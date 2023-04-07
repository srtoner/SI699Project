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

# + endofcell="--"

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
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

torch.set_default_dtype(torch.float32)

from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} \
         if (device == "cuda:0" or device == 'mps') else {}
collate_func = lambda x: tuple(x_.to(device) for x_ in default_collate(x)) \
               if device != "cpu" else default_collate

with open('embedding_data.pkl', 'rb') as f:
    embed = pkl.load(f)

embed_df = pd.DataFrame(embed)


n_classes = embed_df.author_id.nunique()

from sklearn.preprocessing import OneHotEncoder
label_encoder=OneHotEncoder(sparse_output=False)

# -


# X = embed_df['vectors'] # Word Embeddings

# # +
test_size = 0.2
val_size = 0.2
random_state =699


# # +
# with open('embedding_data.pkl', 'rb') as f:
#     embed = pkl.load(f)

# embed_df = pd.DataFrame(embed)

data = U.load_file('data_vFF.pkl', 'pkl', config['DATADIR'])

# --

# +

embed_df = pd.DataFrame(data)

embed_df['join_text'] = embed_df['text'].apply(' '.join)
embed_df = embed_df[['author_id', 'text', 'join_text', 'passage_key']]



# +

n_classes = embed_df.author_id.nunique()

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
y= label_encoder.fit_transform(embed_df['author_id'].to_numpy(dtype='int32').reshape(-1,1))
X = embed_df['passage_key']

X_train, X_test, y_train, y_test = U.train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y)

# Split train set into train and validation sets
X_train, X_val, y_train, y_val = U.train_test_split(X_train, y_train, test_size=val_size/(1-test_size),
                                                    random_state=random_state,
                                                    stratify=y_train)




# +
embed_df['label'] = y

train_set = set(X_train)
val_set = set(X_val)
test_set = set(X_test)
# -

train = embed_df[embed_df['passage_key'].apply(lambda S: S in train_set)]
val = embed_df[embed_df['passage_key'].apply(lambda S: S in val_set)]
test = embed_df[embed_df['passage_key'].apply(lambda S: S in test_set)]

train.to_csv('train.csv', index=False)
val.to_csv('val.csv', index=False)
test.to_csv('test.csv', index=False)

train

from datasets import Dataset

ds = {
      'train' :  Dataset.from_csv('train.csv'),
      'val' :  Dataset.from_csv('val.csv'),
      'test' :  Dataset.from_csv('test.csv'),
      }

embed_df.columns

type(embed_df.join_text.iloc[0])

# +
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers.integrations import TensorBoardCallback
from transformers.trainer_callback import EarlyStoppingCallback
from torch.utils.data import DataLoader

BASE_MODEL = "allenai/longformer-base-4096"
# BASE_MODEL = "lreN5bs16" # Learning Rate 2e-5, batch size 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 400
BATCH_SIZE = 16
EPOCHS = 3

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, 
                                                           num_labels=n_classes,
                                                           ignore_mismatched_sizes=True)

def preprocess_function(examples, test = False):
    # if not test:
    label = examples["label"] 
    examples = tokenizer(examples["text"],
                        truncation=True, 
                        padding="max_length",
                        max_length=MAX_LENGTH,
                        return_tensors='pt')
    
    for key in examples:
        examples[key] = examples[key].squeeze(0)
    # if not test:
    examples["label"] = torch.FloatTensor([label])
    examples = examples.to(device)
    return examples

for split in ds:
    ds[split] = ds[split].map(preprocess_function, 
                                remove_columns=['author_id', 'text', 'passage_key','join_text', 'label'])

    ds[split].set_format('pt')


# +
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="output/",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps = 1,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="f1",
    greater_is_better=True,
    load_best_model_at_end=True,
    weight_decay=0.01,
    report_to = 'tensorboard'
)

early_stop = EarlyStoppingCallback(1, 0.01)
tb = TensorBoardCallback()


from transformers import Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    


# +
from sklearn.metrics import f1_score

def compute_metrics_for_classification(eval_pred):
    predictions, labels = eval_pred
    labels = labels.reshape(-1, 1)
    f1 = f1_score(labels, predictions, squared=True)
    print(f"F1: {f1}")
    
    return {"F1": f1}


# -

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    compute_metrics=compute_metrics_for_classification,
    callbacks=[tb, early_stop]
)

ds['train']

trainer.train()
