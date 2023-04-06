import pickle as pkl
import pandas as pd
import os
import numpy as np
from collections import Counter
import json
import multiprocessing

with open('config.json', 'r') as f:
    config = json.load(f)
cwd = os.getcwd()
os.chdir(config['REPODIR'])
import Utils as U
from Corpus import Corpus
os.chdir(cwd)

