import random
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

random.seed(2)

def init_space(n_points=20, dim=40):
    space = []
    for i in range(n_points):
        v = [random.gauss(0, 1) for z in range(dim)] 
        space.append(v)

    return space

def closest_point(q, space):
    scores = cosine_similarity([space[q]], space)
    scores[0][q] = 0
    return np.argmax(scores)

space = init_space(n_points=1000)
closest = closest_point(0, space)
print(closest)


