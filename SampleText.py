from gutenbergpy.gutenbergcache import GutenbergCache, GutenbergCacheTypes
import gutenbergpy.textget
import os
import json
import pandas as pd
import numpy as np
import pickle as pkl
import time

# Constants
NUM_AUTHORS = 50
WORKS_PER_AUTHOR = 5
CHUNKS_PER_WORK = 10
CHUNK_LENGTH = 50

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

    sample_lines = [chunk * sample_length + original_len - 
                    len(trimmed_book) for chunk in sample_chunks]
    return samples, sample_lines

if __name__ == "__main__":

    # Choose one below
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

    

    author_to_id = {a[1]:a[0] for a in allAuthors}
    id_to_author = {a[0]:a[1] for a in allAuthors}

    with open("sampled_works.json", 'r') as f:
        sampled_works = json.load(f)

    # Container for all samples
    # JSON Format: {author_id, author_name, }
    data = []

    for author_id, works in sampled_works.items():
        # time.sleep(60)
        author_name = id_to_author[int(author_id)]
        for work in works:
            book_id = work[0]; book_title = work[1]; gb_id = work[2]
            try:
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
            except:
                print("Pause")
                continue

    print("Pause")

    with open("data.pkl", "wb") as outfile:
        pkl.dump(data, outfile)