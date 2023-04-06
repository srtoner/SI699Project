# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
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

# +
import pandas as pd

# Load the CSV file into a pandas dataframe

df = U.load_file('data_w_subj.csv', 'csv', config['DATADIR'])
# df = pd.read_csv('/Users/jeffereyreng/Desktop/SI_699/final_project/data/data_w_subj.csv')

# Select only the "author_name", "text", and "subjects" columns
new_df = df[['book_id', 'text', 'subjects']]

# Save the new dataframe to a CSV file
new_df.to_csv('text_subj.csv', index=False)
new_df

# +
import pandas as pd
from sklearn.utils import shuffle

# Load the CSV file into a pandas dataframe
df = pd.read_csv('text_subj.csv', usecols=['book_id', 'text', 'subjects'])

# Shuffle the dataframe
seed = 42  # set the random seed for reproducibility
df = shuffle(df, random_state=seed)

# Print the first few rows of the shuffled dataframe
df.head()

# -

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
y_actual= label_encoder.fit_transform(df['subjects'])
df['subjects']=y_actual

from sklearn.feature_extraction.text import CountVectorizer
BOW = CountVectorizer()
BOW_transformation = BOW.fit_transform(df['text'])

from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_ngram(n_gram,X_train=df['text']):
    vectorizer = TfidfVectorizer(ngram_range=(n_gram,n_gram))
    x_train_vec = vectorizer.fit_transform(X_train)
    return x_train_vec


# Applying tfidf with 1-gram, and 2-gram
tfidf_1g_transformation= tfidf_ngram(1,X_train=df['text'])
tfidf_2g_transformation= tfidf_ngram(2,X_train=df['text'])

#Import packages
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


# +

# Tokenization of each document
tokenized_doc = []
for d in df['text']:
    tokenized_doc.append(word_tokenize(d.lower()))

# Convert tokenized document into gensim formated tagged data
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
tagged_data[0]
# -

# Train doc2vec model
model = Doc2Vec(tagged_data, vector_size=50, window=2, min_count=1, workers=4, epochs = 100)

# import pandas as pd
# from sklearn.utils import shuffle
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from nltk.tokenize import word_tokenize
#
# # Load the CSV file into a pandas dataframe and shuffle the rows
# df = pd.read_csv('text_subj.csv', usecols=['book_id', 'text', 'subjects'])
# df = shuffle(df, random_state=42)
#
# # Train a Doc2Vec model on the text data
# documents = [TaggedDocument(word_tokenize(row.lower()), [i]) for i, row in enumerate(df['text'])]
# model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4, epochs=10)
#
# def get_doc2vec_vector(df):
#     doc2vec_vectors=[]
#     for sentence in df['text']:
#         doc2vec_vectors.append(model.infer_vector(word_tokenize(sentence.lower())))
#     return doc2vec_vectors
# doc2vec_vectors=get_doc2vec_vector(df['text'])
# len(doc2vec_vectors)


# +
def get_doc2vec_vector(df):
    doc2vec_vectors = []
    for sentence in df['text']:
        doc2vec_vectors.append(model.infer_vector(word_tokenize(sentence.lower())))
    return doc2vec_vectors

# Generate Doc2Vec vectors for the "text" column of the dataframe
doc2vec_vectors = get_doc2vec_vector(df)

# Print the number of generated vectors
print(len(doc2vec_vectors))

# +
# pip install transformers
# -

# Check the GPU
import torch
# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# +
# Import Libraries
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Create sentence and label lists
sentences = df['text'].values

# We need to add special tokens at the beginning and end of each sentence for BERT to work properly
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.subjects.values

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Tokenize the sentences and put them in the list tokenized_texts
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway.
# In the original paper, the authors used a length of 512.
MAX_LEN = 128
# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []
hidden_states=[]
# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
# Conver the ids into a tensor representation
batch_size = 4
input_tensor = torch.tensor(input_ids)
masks_tensor = torch.tensor(attention_masks)
train_data = TensorDataset(input_tensor, masks_tensor)
dataloader = DataLoader(train_data, batch_size=batch_size)
# Initialize the model
if torch.cuda.is_available():
    model = BertModel.from_pretrained("bert-base-uncased",output_hidden_states = True,).to('cuda')
else:
    model = BertModel.from_pretrained("bert-base-uncased",output_hidden_states = True,).to('cpu')
model.eval()
outputs = []
# -

for input, masks in dataloader:
    torch.cuda.empty_cache() # empty the gpu memory
    # Transfer the batch to gp
    if torch.cuda.is_available():
        input = input.to('cuda')
        masks = masks.to('cuda')
    # Run inference on the batch
    output = model(input, attention_mask=masks)
    # Transfer the output to CPU again and convert to numpy
    output = output[0].cpu().detach().numpy()
    # Store the output in a list
    outputs.append(output)
# Concatenate all the lists within the list into one list
outputs = [x for y in outputs for x in y]

bert_vectors=np.array(outputs)
bert_vectors=bert_vectors.mean(axis=1)
bert_vectors.shape

import gensim.downloader as api
def get_vectors_pretrained(df, model):
    embedding_vectors = []
    for partition in df['text']:
        sentence = []
        for word in partition.split(' '):
            try:
                sentence.append(model[word])
            except:
                pass
        sentence = np.array(sentence)
        sentence = sentence.mean(axis=0)
        embedding_vectors.append(sentence)
    embedding_vectors = np.array(embedding_vectors)
    return embedding_vectors


# ## GLove

import gensim.downloader as api
glove_model = api.load("glove-wiki-gigaword-300")  # load glove vectors
glove_vectors=get_vectors_pretrained(df,glove_model)
glove_vectors

# ## Fast text

import gensim.downloader as api
fast_text_model = api.load("fasttext-wiki-news-subwords-300")  # load glove vectors
fast_text_vectors=get_vectors_pretrained(df,fast_text_model)
fast_text_vectors

# ## Word2vec

import gensim.downloader as api
word2vec_model = api.load("word2vec-google-news-300")  # load glove vectors
word2vec_vectors = get_vectors_pretrained(df,word2vec_model)
word2vec_vectors

# ## LDA

# +
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import matplotlib.pyplot as plt
import gensim

paragraphs = df["text"].to_list()
docs = []

for sen in paragraphs:
    docs.append(list(sen.split()))
print(len(docs))

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.8)

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]
print(len(corpus[2]))
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

# +
# Set training parameters.
num_topics = 5
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token
#print(len(dictionary))
model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

top_topics = model.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

# +
all_topics = model.get_document_topics(corpus)
num_docs = len(all_topics)

all_topics_csr = gensim.matutils.corpus2csc(all_topics)
lda_to_cluster = all_topics_csr.T.toarray()
lda_to_cluster.shape
# -

from gensim.models.coherencemodel import CoherenceModel
## Evaluating coherence of gensim LDA model
cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
coherence_score = cm.get_coherence()
print(coherence_score)


text_embedding={
    'BOW':BOW_transformation.toarray(),
    'TF_IDF 1_gram':tfidf_1g_transformation.toarray(),
    'Doc2vec':np.array(doc2vec_vectors),
    'Glove':glove_vectors,
    'FastText':fast_text_vectors,
    'Word2vec':word2vec_vectors,
    'BERT':bert_vectors,
    'LDA':lda_to_cluster,
}

import pickle
a_file = open("EmbeddingText_edited.pkl", "wb")
pickle.dump(text_embedding, a_file)
a_file.close()
print('Saved')

import pickle
import pandas as pd
import numpy as np
with open('EmbeddingText_edited.pkl', 'rb') as f:
    text_embedding = pickle.load(f)
y_actual=list(pd.read_csv('text-subj.csv')['subjects'])

# +
import plotly.express as px
from sklearn.decomposition import PCA
pca=PCA(n_components=2,)
embedding=text_embedding.copy()

for key in embedding.keys():
    # embedding[key]=pca.fit_transform(embedding[key])
    embedding[key]=pca.fit_transform(embedding[key])
    df=pd.DataFrame({'PCA1':embedding[key][:,0],'PCA2':embedding[key][:,1],'Target':y_actual})
    fig = px.scatter(data_frame =df, x='PCA1', y='PCA2', color='Target')
    fig.update_layout(title={'text':f'{key}','x':0.5},height=500,width=700)
    fig.show()
# -



# +
import pandas as pd

df = pd.read_csv('data_w_subj.csv')


# +
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the CSV file
df = pd.read_csv('text_subj.csv')

# Extract the text column
text_data = df['text']

# Load the pre-trained GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Load the pre-trained GPT-2 model
model = AutoModelForCausalLM.from_pretrained('gpt2')

# Set the device to use (either GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Generate text based on a prompt
prompt = 'The book was about'
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
output_ids = model.generate(input_ids, max_length=100, do_sample=True)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(generated_text)


# +
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextGenerationPipeline

# Load the CSV file
df = pd.read_csv('text_subj.csv')

# Extract the text column
text_data = df['text']

# Load a pre-trained language model and tokenizer
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fine-tune the model on your specific dataset
# ...

# Customize the model architecture
# ...

# Set up a pipeline for text generation
generator: TextGenerationPipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Generate text based on a prompt
prompt = 'The book was about'
generated_text = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']



# +
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

# +
import pandas as pd
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextGenerationPipeline, AutoModelForSequenceClassification

# Load the CSV file
df = pd.read_csv('text_subj.csv')

# Extract the text column
text_data = df['text']

# Load a pre-trained language model and tokenizer
model_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
classification_model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Fine-tune the model on your specific dataset
# ...

# Customize the model architecture
# ...

# Set up a pipeline for text generation
generator: TextGenerationPipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Generate text based on a prompt
prompt = 'The book was about'
generated_text = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']

# Apply NLP techniques to the generated text
nlp = spacy.load('en_core_web_sm')
doc = nlp(generated_text)

# Extract named entities from the generated text
for ent in doc.ents:
    print(ent.text, ent.label_)

# Extract parts of speech from the generated text
for token in doc:
    print(token.text, token.pos_)

# Apply sentiment analysis to the generated text using the BERT model
sentiment_analyzer = pipeline('sentiment-analysis', model=classification_model, tokenizer=tokenizer)
sentiment = sentiment_analyzer(generated_text)[0]
print('Sentiment:', sentiment['label'], sentiment['score'])

# Generate more text based on the previous output and repeat the process
# ...


# +
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load the CSV file
df = pd.read_csv('text_subj.csv')

# Extract the text column
text_data = df['text']

# Load a pre-trained language model and tokenizer
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fine-tune the model on your specific dataset
train_texts = text_data.values.tolist()
train_encodings = tokenizer(train_texts, truncation=True, padding=True)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = MyDataset(train_encodings)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy = "epoch",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    logging_steps=5000,
    save_steps=5000,
    warmup_steps=1000,
    learning_rate=5e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Customize the model architecture
# ...

# -

#

# +
# pip install textblob

# +
# Apply sentiment analysis to the generated text

from textblob import TextBlob

# Apply sentiment analysis to the generated text
blob = TextBlob(generated_text)
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity

print('Sentiment:', polarity)
print('Subjectivity:', subjectivity)


# +
#Generate more text based on the previous output and repeat the process
# Set up a loop to generate multiple outputs
import spacy
nlp = spacy.load("en_core_web_sm")

for i in range(3):
    # Generate text based on the previous output
    prompt = generated_text
    generated_text = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']

    # Apply NLP techniques to the generated text
    doc = nlp(generated_text)
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    parts_of_speech = [(token.text, token.pos_) for token in doc]

    # Apply sentiment analysis to the generated text
    blob = TextBlob(generated_text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Print out the generated text and its properties
    print('Generated text:', generated_text)
    print('Named entities:', named_entities)
    print('Parts of speech:', parts_of_speech)
    print('Sentiment:', polarity)
    print('Subjectivity:', subjectivity)
    print('---')

# -


