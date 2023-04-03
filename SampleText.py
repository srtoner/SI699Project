from gutenbergpy.gutenbergcache import GutenbergCache, GutenbergCacheTypes
import gutenbergpy.textget
import os
import json
import pandas as pd
import numpy as np
import pickle as pkl
import time

from collections import Counter, defaultdict
import itertools

# Constants
NUM_AUTHORS = 50
WORKS_PER_AUTHOR = 5
CHUNKS_PER_WORK = 30
CHUNK_LENGTH = 50

with open('config.json', 'r') as f:
    config = json.load(f)
cwd = os.getcwd()
os.chdir(config['REPODIR'])
import Utils as U
from Corpus import Corpus
os.chdir(cwd)

np.random.seed(699)
cache  = GutenbergCache.get_cache()

def fetch_book(id):
    raw_book = gutenbergpy.textget.get_text_by_id(id)
    clean_book = gutenbergpy.textget.strip_headers(raw_book)
    return clean_book, raw_book

def trim_book(clean_book, trim_pct = 0.1):
    """
        Removes the first and last "trim %" of the book to 
        prevent sampling the table of contents
    """

    lines = clean_book.strip().decode('utf-8').splitlines()
    n = len(lines)
    start = n * trim_pct // 1; end = n * (1 - trim_pct) // 1

    return n, lines[int(start):int(end)]

def partition_and_sample(trimmed_book, n_samples, sample_length, original_len):
    K = len(trimmed_book) // sample_length # Number of partitions
    sample_chunks = np.random.choice(K, min(K, n_samples), replace = False)
    samples = []
    for chunk in sample_chunks:
        samples.append(
            trimmed_book[(chunk * sample_length):((chunk + 1) * sample_length)]
            )

    sample_lines = [chunk * sample_length + original_len - len(trimmed_book) 
                    for chunk in sample_chunks]
    return samples, sample_lines

if __name__ == "__main__":

    # For prior to april workflow
    # catalog = pd.read_csv("pg_catalog.csv")
    # catalog = catalog[catalog["Type"] == "Text"]
    # catalog = catalog[catalog["Language"] == "en"]
    # filtered_ids = str(tuple(catalog['Text#']))

    catalog = U.load_file('seed_works.csv', 'csv', config['DATADIR'])
    works = tuple(x.replace('PG','') for x in catalog.id)
    catalog['gbid'] = catalog.id.apply(lambda x: int(x.replace('PG', '')))
    filtered_ids = str(works)

    allAuthors = [s for s in cache.native_query(
    "SELECT a.authorid, a.name, count(b.id) as book_count from \
    (SELECT * from authors \
    LEFT JOIN book_authors \
    ON id = authorid) as a \
    LEFT JOIN \
    (SELECT * FROM books \
    WHERE gutenbergbookid IN {}) as b \
    ON b.id = a.bookid \
    GROUP BY a.id, a.name \
    HAVING book_count >= 5 \
    ORDER BY book_count DESC;".format(filtered_ids))]

    book_ids = [s for s in cache.native_query(
    "SELECT id, gutenbergbookid FROM books \
    WHERE gutenbergbookid IN {};".format(filtered_ids))]



    author_to_id = {a[1]:a[0] for a in allAuthors}
    id_to_author = {a[0]:a[1] for a in allAuthors}

    works = U.load_file('seed_works.csv', 'csv', config['DATADIR'])
    works.id = works.id.apply(lambda X: X[2:])
    works = works.to_dict(orient='records')
    
    gib2bid = {b_map[1]:b_map[0] for b_map in book_ids}
    data = []

    if os.path.exists('data_vTemp.pkl'):
        with open('data_vTemp.pkl', 'rb') as f:
            data = pkl.load(f)
    # data_df = pd.DataFrame(data).drop_duplicates()
    processed_ids = set([d['gutenbergbookid'] for d in data])
    # for author_id, works in sampled_works.items():
    #     # time.sleep(60)
    #     author_name = id_to_author[int(author_id)]
    for work in works:
        if work['id'] in processed_ids:
            continue
        # book_id = work[0]; book_title = work[1]; gb_id = work[2]
       
        try:
            book_id = gib2bid[int(work['id'])]; book_title = work['title']; gb_id = int(work['id'])
            author_id = author_to_id[work['author']]; author_name = work['author']
        
   
            clean_book, _ = fetch_book(book_id) # Throws an error when not a text
            original_len, trimmed_book = trim_book(clean_book)
            excerpts, excerpt_lines = partition_and_sample(
                                    trimmed_book, 
                                    CHUNKS_PER_WORK, 
                                    CHUNK_LENGTH, 
                                    original_len
                                    )
            for idx, passage in enumerate(excerpts):
                data.append({
                    "author_id":author_id,
                    "author_name":author_name,
                    "book_id":book_id,
                    "gutenbergbookid":gb_id,
                    "title":book_title,
                    "text":passage,
                    "text_lines":excerpt_lines[idx]
                })
                processed_ids.update([work['id']])
        except:
            print("Pause")
            continue

    # Add in additional subject data
    dataset_subjects = Counter()

    test = pd.DataFrame(data)
    out = test.merge(catalog, how='left', left_on='gutenbergbookid', right_on='gbid')
    

    # # for d in test:
    # for d in data:
    #     temp_subjects = catalog[catalog['Text#'] == d['gutenbergbookid']]['Subjects'].to_string(index = False).split(";")
    #     d['subjects'] = [dat.strip() for dat in list(itertools.chain.from_iterable([temp.strip().split("--") for temp in d['subjects']]))]
    #     dataset_subjects.update(d['subjects'])

    with open("data_vFinal.pkl", "wb") as outfile:
        pkl.dump(out.to_dict(orient='records'), outfile)