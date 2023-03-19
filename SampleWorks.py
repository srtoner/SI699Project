from gutenbergpy.gutenbergcache import GutenbergCache, GutenbergCacheTypes
import os
import json
import pandas as pd
import numpy as np
import pickle as pkl

# Constants
NUM_AUTHORS = 50
WORKS_PER_AUTHOR = 5
CHUNKS_PER_WORK = 10
CHUNK_LENGTH = 50

cache  = GutenbergCache.get_cache()

allAuthors = [s for s in cache.native_query(
    "SELECT a.authorid, a.name, count(b.id) as book_count from \
    (SELECT * from authors \
    LEFT JOIN book_authors \
    ON id = authorid) as a \
    LEFT JOIN \
    (SELECT * FROM books \
    WHERE languageid = 1) as b \
    ON b.id = a.bookid \
    GROUP BY a.id, a.name \
    HAVING book_count >= 5 \
    ORDER BY book_count DESC;")]

pd.DataFrame(allAuthors).to_csv("all_authors.csv")

np.random.seed(699)

author_ids = [a[0] for a in allAuthors]

sampled_authors = np.random.choice(author_ids, size=NUM_AUTHORS)
id_author_map = {a[0]:a[1] for a in allAuthors}

# Sample Book Ids for Each author
book_sets = []
for author in sampled_authors:
    book_set = [b[0] for b in cache.native_query(
        "SELECT DISTINCT ba.bookid FROM (SELECT bookid FROM book_authors WHERE authorid = {}) AS ba \
         INNER JOIN (SELECT * FROM books WHERE languageid = 1) as b \
         ON b.id = ba.bookid;".format(author)
    )]
    book_sets.append(book_set)

author_works = {}

for author_idx, book_set in enumerate(book_sets):
    sampled_works = np.random.choice(book_set, WORKS_PER_AUTHOR, replace = False)
    book_info = [info for info in cache.native_query(
        "SELECT b.id, t.name, b.gutenbergbookid FROM (SELECT DISTINCT id, gutenbergbookid FROM books  \
        WHERE id in {} ) as b \
        LEFT JOIN titles AS t \
        ON t.bookid = b.id;".format(str(tuple(sid for sid in sampled_works))))]
    author_works[int(sampled_authors[author_idx])] = book_info

with open("sampled_works.json", 'w') as f:
    json.dump(author_works, f)