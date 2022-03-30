import random
import numpy as np

np.random.seed(2)

def lsh(space, n_sets=10, n_hyp=5):
    space = np.array(space)
    tables = []
    planes = []
    for s in range(n_sets):
        hash_table = {}
        plane_norms = np.random.rand(n_hyp, space.shape[1]) - 0.5
        for i, d in enumerate(space):
            _hash = ''.join((d @ plane_norms.T > 0).astype(int).astype(str))
            if _hash not in hash_table.keys(): hash_table[_hash] = []
            hash_table[_hash].append(i)

        tables.append(hash_table)
        planes.append(plane_norms)

    return tables, planes

def predict(i, space, tables, planes):
    for s in range(len(tables)):
        hash_table = tables[s]
        plane_norms = planes[s]
        _hash = ''.join((space[i] @ plane_norms.T).astype(int).astype(str))
        print(_hash)








