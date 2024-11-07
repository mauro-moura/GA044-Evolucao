
import random

import numpy as np

def genetic_drift(population, seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    population = np.array(population)
    new_pop = np.empty(len(population), dtype=population.dtype)
    for i in range(len(population)):
        new_pop[i] = np.random.choice(population)
    return new_pop

