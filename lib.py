import tqdm
import numpy as np

from typing import Union

from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

class RandomProjectionLSH(object):
    def __init__(self, 
                 vec_space_path: str,
                 n_sets: int,
                 n_bits: int,
                 load_dir: Union[str, None] = None,
                 save_dir: Union[str, None] = 'hash_table'):
        self.n_sets = n_sets
        self.n_bits = n_bits

        self.space = KeyedVectors.load_word2vec_format(vec_space_path, binary=True)
        
        self.tables, self.planes = [], []
        if load_dir is not None:
            self.tables, self.planes = self.load_hash_tables()

        self.save_dir = save_dir 

    def load_hash_tables(self):
        self.tables = pickle.load(open(f"{load_dir}/tables.pkl"))
        self.planes = pickle.load(open(f"{load_dir}/planes.pkl"))

    def build_hash_tables(self): 
        for s in tqdm.tqdm(range(self.n_sets), desc="sets"):
            hash_table = {}
            plane_norms = np.random.rand(self.n_bits, self.space.vector_size) - 0.5
            for idx, word in tqdm.tqdm(enumerate(list(self.space.vocab.keys())), desc="words"):
                _hash = ''.join((self.space[word] @ plane_norms.T > 0).astype(int).astype(str))
                if _hash not in hash_table.keys(): hash_table[_hash] = []
                hash_table[_hash].append(word)

            self.tables.append(hash_table)
            self.planes.append(plane_norms)

        if self.save_dir is not None:
            pickle.dump(self.tables, open(f"{save_dir}/tables.pkl", 'wb'))
            pickle.dump(self.planes, open(f"{save_dir}/planes.pkl", 'wb'))

    def search(query:str, k:int=10): 
        neighbours = []
        for hash_table, plane_norms in zip(self.tables, self.planes):
            _hash = ''.join((self.space[query] @ plane_norms.T > 0).astype(int).astype(str))
            neighbours.extend([e for e in hash_table[_hash] if e not in neighbours])

        sims = cosine_similarity([self.space[query]], self.space[neighbours])[0]
        scores = { k: v for k,v in zip(neighbours, sims) }
        del scores[query]

        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]) 





