import os
import pickle

import numpy as np

from tqdm import tqdm
from typing import Union
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

class RandomProjectionLSH(object):
    def __init__(self, 
                 vec_space_path: str,
                 n_sets: int = 10,
                 n_bits: int = 10,
                 load_dir: Union[str, None] = None,
                 save_dir: Union[str, None] = 'hash_table'):
        self.n_sets = n_sets
        self.n_bits = n_bits

        self.space = KeyedVectors.load_word2vec_format(vec_space_path, binary=True)

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.tables, self.planes = [], []
        if load_dir is not None: 
            self.load_hash_tables()
        else: 
            self.build_hash_tables()
            if save_dir is not None:
                self.save_hash_tables()

    def load_hash_tables(self):
        print(f'loading from dir.. {self.load_dir}')
        self.tables = pickle.load(open(f"{self.load_dir}/tables.pkl", 'rb'))
        self.planes = pickle.load(open(f"{self.load_dir}/planes.pkl", 'rb'))
        print('successfully loaded!')

    def save_hash_tables(self):
        print(f'saving to dir.. {self.save_dir}')
        if not os.path.exists(self.save_dir): os.mkdir(self.save_dir)
        pickle.dump(self.tables, open(f"{self.save_dir}/tables.pkl", 'wb'))
        pickle.dump(self.planes, open(f"{self.save_dir}/planes.pkl", 'wb'))
        print('successfully saved!')

    def build_hash_tables(self): 
        for s in tqdm(range(self.n_sets), desc="sets"):
            hash_table = {}
            vecs = self.space.vectors[np.random.randint(len(self.space.vectors), size=(2 * self.n_bits))]
            plane_norms = vecs[1::2] - vecs[::2] 
            for _id in tqdm(list(self.space.vocab.keys()), desc="ids"):
                _hash = ''.join((self.space[_id] @ plane_norms.T > 0).astype(int).astype(str))
                if _hash not in hash_table.keys(): hash_table[_hash] = []
                hash_table[_hash].append(_id)

            self.tables.append(hash_table)
            self.planes.append(plane_norms)

    def search(self, query:str, k:int=10): 
        neighbours = []
        for hash_table, plane_norms in tqdm(zip(self.tables, self.planes), total=len(self.tables)):
            _hash = ''.join((self.space[query] @ plane_norms.T > 0).astype(int).astype(str))
            neighbours.extend(hash_table[_hash])

        neighbours = list(set(neighbours))
        sims = cosine_similarity([self.space[query]], self.space[neighbours])[0]
        scores = { k: v for k,v in zip(neighbours, sims) }
        del scores[query]

        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]) 





