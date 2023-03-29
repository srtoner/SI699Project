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

import pandas as pd
import os
from collections import Counter, defaultdict
import json
import itertools

# load config
with open('config.json', 'r') as f:
    config = json.load(f)
cwd = os.getcwd()
os.chdir(config['REPODIR'])
import utils.Utils as U
from Corpus import Corpus
os.chdir(cwd)

data = U.load_file('data_w_subj_new.csv', 'csv', config['DATADIR'])
data

subjects = data['subject'].unique()

n_subjects = 11

# +
import re

# Remove punctuation
data['text_processed'] = data['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Convert to lowercase
data['text_processed'] = data['text_processed'].apply(lambda x: x.lower())

data.head()

# +
# from wordcloud import WordCloud

# # Join together
# long_string = ','.join(list(data['text_processed'].values))

# # Generate a wordCloud
# wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# wordcloud.generate(long_string)

# # Visualize
# wordcloud.to_image()
# -

# Get number of unique subjects
num_subjects = data['subjects'].nunique()
num_subjects = n_subjects



# +
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score



# Split data into train/validation/test sets with a ratio of 70/15/15
train, test = train_test_split(data, test_size=0.3, random_state=42)
val, test = train_test_split(test, test_size=0.5, random_state=42)

# Preprocess text data - remove stop_words
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train['text_processed'])
X_val = vectorizer.transform(val['text_processed'])
X_test = vectorizer.transform(test['text_processed'])

tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, min_df=2, stop_words="english"
)

# Train LDA model
lda = LatentDirichletAllocation(n_components=n_subjects) 
                                #,max_iter=50, learning_method='online', random_state=42
lda.fit(X_train)

# Predict subject for documents in validation set
y_val = lda.transform(X_val)

# Predict subject for documents in test set
y_test = lda.transform(X_test)

# Convert subject column to numerical labels
labels_train = pd.factorize(train['subject'])
labels_val = pd.factorize(val['subject'])
labels_test = pd.factorize(test['subject'])


# This is probably a better way of getting the classifier
# accuracy_score(labels_val[0], np.apply_along_axis(np.argmax, 1, y_val))



# Train a classifier on the LDA topics
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(y_val, labels_val)

# Evaluate performance on the validation set
pred_val = clf.predict(y_val)
acc_val = accuracy_score(labels_val, pred_val)
print('Accuracy on validation set:', acc_val)

# Evaluate performance on the test set
pred_test = clf.predict(y_test)
acc_test = accuracy_score(labels_test, pred_test)
print('Accuracy on test set:', acc_test)
# -


