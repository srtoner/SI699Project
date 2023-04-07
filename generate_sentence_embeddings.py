import gensim.downloader as api
from multiprocessing import Pool, cpu_count

# Load pre-trained GloVe embeddings
glove_model = api.load('glove-wiki-gigaword-300')

# Define the function to generate sentence embeddings
def generate_embedding(sentence):
    words = sentence.lower().split()
    word_embeddings = [glove_model[word] for word in words if word in glove_model]
    if not word_embeddings:
        return None
    sentence_embedding = sum(word_embeddings) / len(word_embeddings)
    return sentence_embedding

# Tokenize and preprocess your corpus
corpus = ["This is the first sentence.", "This is the second sentence.", ...]

# Define the number of processes to use
num_processes = cpu_count()

# Split the corpus into chunks
chunk_size = len(corpus) // num_processes
chunks = [corpus[i:i+chunk_size] for i in range(0, len(corpus), chunk_size)]

# Generate sentence embeddings in parallel
with Pool(num_processes) as pool:
    sentence_embeddings = []
    for chunk_embeddings in pool.imap(map, chunks):
        sentence_embeddings.extend(chunk_embeddings)

# Save sentence embeddings to file
with open('sentence_embeddings.txt', 'w') as f:
    for embedding in sentence_embeddings:
        f.write(' '.join(str(x) for x in embedding) + '\n')