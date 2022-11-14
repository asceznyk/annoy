import sys
import time
import random
import tracemalloc
import numpy as np

import pickle

from sklearn.metrics.pairwise import cosine_similarity

from lsh import *

def main(vec_space_path:str, load_path:str, query:str, top_k:int=7):
    print(f"finding points closest to {query} ...")

    space = models.KeyedVectors.load_word2vec_format(vec_space_path, binary=True) 
    tables = pickle.load(open(f"{load_path}/tables.pkl"))
    planes = pickle.load(open(f"{load_path}/planes.pkl"))

    start = time.time()
    scores = search(query, space, tables, planes, k=top_k)
    print(time.time() - start)

    for word, score in scores.items():
        print(f"word: {word} -> similarity score: {score}")
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='search function on query')
    parser.add_argument("--vec_space_path", type=str, required=True, help='vector space to search from')
    parser.add_argument("--load_dir", type=str, default='./hash_tables', help="directory for loading hyperlpanes and hash tables..")
    parser.add_argument("--query", type=str, default='', help="query for search")
    
    to_args = {}
    args = parser.parse_args()
    for k in args.__dict__:
        to_args[k] = args.__dict__[k]
    main(**to_args)


