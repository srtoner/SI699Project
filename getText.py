import gutenbergpy as gbp
import gutenbergpy.textget
import os
import argparse
import sys

import pandas as pd

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

    return lines[int(start):int(end)]

if __name__ == "__main__":
    clean, raw = fetch_book(2701) # Moby Dick
    trimmed = trim_book(clean)
    print("Pause")