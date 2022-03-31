import time
import random
import tracemalloc
import numpy as np

import pickle 

from sklearn.metrics.pairwise import cosine_similarity

from lsh import *

def init_space(n_points=20, dim=40):
    space = []
    for i in range(n_points):
        v = [random.gauss(0, 1) for z in range(dim)] 
        space.append(v)

    return space

space = init_space(n_points=10000)
tables, planes = lsh(space) ##this will take time!

pickle.dump(space, open('space.pkl', 'wb'))
pickle.dump(tables, open('tables.pkl', 'wb'))
pickle.dump(planes, open('planes.pkl', 'wb'))




