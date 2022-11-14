import time
import random
import pickle
import argparse
import tracemalloc

import numpy as np

from gensim import models
from sklearn.metrics.pairwise import cosine_similarity

from lsh import *

def main(vec_space_path:str, n_sets:int = 100, n_bits:int = 10, save_dir:str = './hash_tables'):
    space = models.KeyedVectors.load_word2vec_format(vec_space_path, binary=True)
    tables, planes = build_hash_tables(space) 
    pickle.dump(tables, open(f"{save_dir}/tables.pkl", 'wb'))
    pickle.dump(planes, open(f"{save_dir}/planes.pkl", 'wb'))
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='building hash tables for LSH..')
    parser.add_argument("--vec_space_path", type=str, required=True, help='load vector space from this path, usually a .bin file')
    parser.add_argument("--n_sets", type=int, default=10, help="no. of hash tables")
    parser.add_argument("--n_bits", type=int, default=10, help="n-bits for hashes")
    parser.add_argument("--save_dir", type=str, default='./hash_tables', help="directory for saving hyperlpanes and hash tables..")
    
    to_args = {}
    args = parser.parse_args()
    for k in args.__dict__:
        to_args[k] = args.__dict__[k]
    main(**to_args)



