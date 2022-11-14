import numpy as np

from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

def build_hash_tables(space:Word2VecKeyedVectors, n_sets:int=10, n_bits:int=5): 
    tables = []
    planes = []
    for s in range(n_sets):
        hash_table = {}
        plane_norms = np.random.rand(n_bits, space.vector_size) - 0.5
        for idx, word in enumerate(list(space.vocab.keys())):
            _hash = ''.join((space[word] @ plane_norms.T > 0).astype(int).astype(str))
            if _hash not in hash_table.keys(): hash_table[_hash] = []
            hash_table[_hash].append(word)

        tables.append(hash_table)
        planes.append(plane_norms)

    return tables, planes

def search(query:str, space:Word2VecKeyedVectors, tables:np.array, planes:np.array, k:int=3): 
    neighbours = []
    for hash_table, plane_norms in zip(tables, planes):
        _hash = ''.join((space[query] @ plane_norms.T > 0).astype(int).astype(str))
        neighbours.extend([e for e in hash_table[_hash] if e not in neighbours])

    sims = cosine_similarity([space[query]], space[neighbours])[0]
    scores = { k: v for k,v in zip(neighbours, sims) }
    del scores[query]
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k])
    
    return scores





