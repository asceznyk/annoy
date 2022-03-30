import random
import numpy as np

np.random.seed(2)

def lsh(dataset, n_sets=30, n_hyp=30):
    dataset = np.array(dataset)
    for s in range(n_sets):
        plane_norms = np.random.rand(n_hyp, dataset.shape[1]) - 0.5
        print(plane_norms)












