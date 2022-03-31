import time
import random
import tracemalloc
import numpy as np

import pickle

from sklearn.metrics.pairwise import cosine_similarity

from lsh import *

def closest_point(q, space, k=3):
    scores = {}
    for p in range(len(space)):
        scores[p] = cosine_similarity([space[q]], [space[p]])[0][0]
    scores[q] = 0
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]) 

space = pickle.load(open('space.pkl', 'rb'))
tables = pickle.load(open('tables.pkl', 'rb'))
planes = pickle.load(open('planes.pkl', 'rb'))

top_k = 7
query = 10 ##feel free to explore!

start = time.time()
print(closest_point(query, space, k=top_k))
print(time.time() - start)

start = time.time()
print(predict(query, space, tables, planes, k=top_k))
print(time.time() - start)


