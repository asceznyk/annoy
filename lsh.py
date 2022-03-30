import random
import numpy as np

np.random.seed(2)

def lsh(dataset, n_sets=1, n_hyp=10):
    dataset = np.array(dataset)
    tables = []
    for s in range(n_sets):
        hash_table = {}
        plane_norms = np.random.rand(n_hyp, dataset.shape[1]) - 0.5
        for i, d in enumerate(dataset):
            _hash = ''.join((d @ plane_norms.T > 0).astype(int).astype(str))
            if _hash not in hash_table.keys(): hash_table[_hash] = []
            hash_table[_hash].append(i)

        print(hash_table)













