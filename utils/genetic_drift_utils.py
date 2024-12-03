
import random

import numpy as np
from numba import jit, prange

def mutate_individue(individue, u=0, v=0):
    # Individue is a string "AA" or "aa" or "Aa"
    if individue == "AA" and random.random() < u:
        return "aa"

@jit(nopython=True, parallel=True)
def genetic_drift(population, u=0, v=0, seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    population = np.array(population)
    new_pop = np.empty(len(population), dtype=population.dtype)
    for i in prange(len(population)):
        new_pop[i] = mutate_individue(np.random.choice(population), u, v)
    return new_pop
